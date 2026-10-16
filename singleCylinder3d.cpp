/*  Lattice Boltzmann sample, written in C++, using the OpenLB
 *  library
 *
 *  Single cylinder with periodic spanwise boundaries, symmetry side
 *  boundaries, a pressure outlet and three nested grid-refinement levels.
 */

#include <olb3D.h>
#include <olb3D.hh>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <functors/lattice/latticePhysBoundaryForce3D.h>
#include <functors/lattice/latticePhysPressure3D.h>
#include <functors/analytical/interpolationF3D.h>

using namespace olb;
using namespace olb::descriptors;

using T = FLOATING_POINT_TYPE;
using DESCRIPTOR = D3Q19<>;
using BulkDynamics = SmagorinskyBGKdynamics<T,DESCRIPTOR>;

const T D = 0.05715;
const T inletVelocity = 1.03;
const T rampTime = 5.0;   // inlet velocity ramps from 0 to U0 over this time

IndicatorCuboid3D<T> makeSimulationDomain()
{
  return IndicatorCuboid3D<T>(Vector<T,3>{40.0*D, 3.0*D, 20.0*D},
                              Vector<T,3>{-20.0*D, 0.0, -10.0*D});
}

IndicatorCuboid3D<T> makeFluidDomain(T dx)
{
  return IndicatorCuboid3D<T>(
    Vector<T,3>{40.0*D - 2.0*dx, 3.0*D, 20.0*D - 2.0*dx},
    Vector<T,3>{-20.0*D + dx, 0.0, -10.0*D + dx});
}

std::shared_ptr<IndicatorF3D<T>> makeCylinder()
{
  const T radius = D/2.0 + T(128)*std::numeric_limits<T>::epsilon()*D;
  return std::make_shared<IndicatorCylinder3D<T>>(
    Vector<T,3>{0.0, 1.5*D, 0.0},
    Vector<T,3>{0.0, 1.0, 0.0},
    radius,
    3.0*D);
}

template <typename DECOMPOSITION>
void splitCuboidByLatticeWidths(DECOMPOSITION& decomposition, int iC, int iD,
                                std::vector<int> widths)
{
  const int extent = decomposition.get(iC).getExtent()[iD];
  int fixedWidth = 0;
  for (std::size_t i = 0; i + 1 < widths.size(); ++i) {
    fixedWidth += widths[i];
  }
  widths.back() = extent - fixedWidth;
  if (widths.back() <= 0) {
    throw std::runtime_error("Invalid splitCuboidByLatticeWidths widths");
  }

  std::vector<T> fractions;
  fractions.reserve(widths.size());
  for (std::size_t i = 0; i + 1 < widths.size(); ++i) {
    fractions.emplace_back((T(widths[i]) + T(0.25)) / T(extent));
  }
  fractions.emplace_back(T(0));
  decomposition.splitFractional(iC, iD, fractions);
}

/// Split every cuboid that contains one of the given physical boundaries.
///
/// The lagrava coarse-fine coupling maps fine to coarse cells with local
/// lattice indices (fine index = 2 * coarse index), which requires each fine
/// cuboid to share its physical origin with its coarse parent cuboid. This
/// helper makes the coarse level's cuboid boundaries coincide with the finer
/// level's box boundaries, so that every fine cuboid is an in-place
/// refinement of a coarse cuboid.
template <typename DECOMPOSITION>
void splitCuboidsAtPhysR(DECOMPOSITION& decomposition, int iD,
                         const std::vector<T>& boundaries)
{
  bool splitOccurred = true;
  while (splitOccurred) {
    splitOccurred = false;
    for (int iC = 0; iC < decomposition.size(); ++iC) {
      auto& cuboid = decomposition.get(iC);
      const T origin = cuboid.getOrigin()[iD];
      const T delta = cuboid.getDeltaR();
      const T hi = origin + cuboid.getExtent()[iD]*delta;

      std::vector<int> widths;
      T prev = origin;
      bool anyInside = false;
      for (T b : boundaries) {
        // Only split at a boundary that is at least one lattice cell away
        // from the previous cut and from the far edge.  Otherwise rounding
        // can place the boundary onto the piece edge and produce a
        // degenerate zero-width remainder.
        if (b > prev + delta && b < hi - delta) {
          const int w = std::lround((b - prev) / delta);
          if (w > 0) {
            widths.push_back(w);
            prev += T(w)*delta;
            anyInside = true;
          }
        }
      }
      if (anyInside) {
        widths.push_back(0);  // remainder
        int fixedWidth = 0;
        for (std::size_t i = 0; i + 1 < widths.size(); ++i) {
          fixedWidth += widths[i];
        }
        if (fixedWidth < cuboid.getExtent()[iD]) {
          splitCuboidByLatticeWidths(decomposition, iC, iD, widths);
          splitOccurred = true;
          break;  // splitFractional renumbers cuboids, so restart the scan
        }
        // Degenerate split (boundary on the piece edge): skip this cuboid.
      }
    }
  }
}

/// Split only cuboids lying wholly inside the core region.
template <typename DECOMPOSITION>
void splitCuboidsInCoreAtPhysR(DECOMPOSITION& decomposition, int iD,
                               const std::vector<T>& boundaries,
                               const Vector<T,3>& coreOrigin,
                               const Vector<T,3>& coreExtent)
{
  bool splitOccurred = true;
  while (splitOccurred) {
    splitOccurred = false;
    for (int iC = 0; iC < decomposition.size(); ++iC) {
      auto& cuboid = decomposition.get(iC);
      const T delta = cuboid.getDeltaR();
      const T tol = delta / T(1024);
      const T coreMax = coreOrigin[0] + coreExtent[0];
      const T coreYMax = coreOrigin[1] + coreExtent[1];
      const T coreZMax = coreOrigin[2] + coreExtent[2];

      bool inCore =
        cuboid.getOrigin()[0] >= coreOrigin[0] - tol &&
        cuboid.getOrigin()[0] + (cuboid.getExtent()[0]-1)*delta
          <= coreMax + tol &&
        cuboid.getOrigin()[1] >= coreOrigin[1] - tol &&
        cuboid.getOrigin()[1] + (cuboid.getExtent()[1]-1)*delta
          <= coreYMax + tol &&
        cuboid.getOrigin()[2] >= coreOrigin[2] - tol &&
        cuboid.getOrigin()[2] + (cuboid.getExtent()[2]-1)*delta
          <= coreZMax + tol;
      if (!inCore) {
        continue;
      }

      const T origin = cuboid.getOrigin()[iD];
      const T hi = origin + cuboid.getExtent()[iD]*delta;

      std::vector<int> widths;
      T prev = origin;
      bool anyInside = false;
      for (T b : boundaries) {
        if (b > prev + delta && b < hi - delta) {
          const int w = std::lround((b - prev) / delta);
          if (w > 0) {
            widths.push_back(w);
            prev += T(w)*delta;
            anyInside = true;
          }
        }
      }
      if (anyInside) {
        widths.push_back(0);
        int fixedWidth = 0;
        for (std::size_t i = 0; i + 1 < widths.size(); ++i) {
          fixedWidth += widths[i];
        }
        if (fixedWidth < cuboid.getExtent()[iD]) {
          splitCuboidByLatticeWidths(decomposition, iC, iD, widths);
          splitOccurred = true;
          break;
        }
      }
    }
  }
}

template <typename DECOMPOSITION>
void printCuboidNodeBounds(const std::string& name,
                           const DECOMPOSITION& decomposition)
{
  OstreamManager clout(std::cout, "cuboidBounds");
  for (int iC = 0; iC < decomposition.size(); ++iC) {
    const auto& cuboid = decomposition.get(iC);
    const auto origin = cuboid.getOrigin();
    const auto extent = cuboid.getExtent();
    const T dx = cuboid.getDeltaR();
    Vector<T,3> nodeMax{
      origin[0] + (extent[0] - 1)*dx,
      origin[1] + (extent[1] - 1)*dx,
      origin[2] + (extent[2] - 1)*dx};
    Vector<T,3> nodeCenter{
      T(0.5)*(origin[0] + nodeMax[0]),
      T(0.5)*(origin[1] + nodeMax[1]),
      T(0.5)*(origin[2] + nodeMax[2])};

    clout << name << " cuboid " << iC
          << " origin=" << origin
          << " nodeMax=" << nodeMax
          << " nodeCenter=" << nodeCenter
          << " extent=" << extent
          << " dx=" << dx << std::endl;
  }
}

void printMaterialBounds(SuperGeometry<T,3>& sGeometry,
                         const std::string& name)
{
  OstreamManager clout(std::cout, "materialBounds");
  auto& statistics = sGeometry.getStatistics();
  for (int material : {1, 2, 3, 4, 5, 6}) {
    const auto n = statistics.getNvoxel(material);
    if (n == 0) {
      continue;
    }
    const auto minR = statistics.getMinPhysR(material);
    const auto maxR = statistics.getMaxPhysR(material);
    const auto center = statistics.getCenterPhysR(material);
    clout << name << " material " << material
          << " n=" << n
          << " min=" << minR
          << " max=" << maxR
          << " center=" << center << std::endl;
  }
}

void prepareGeometry(const UnitConverter<T,DESCRIPTOR>& converter,
                     SuperGeometry<T,3>& sGeometry)
{
  OstreamManager clout(std::cout, "prepareGeometry");
  clout << "Prepare Geometry ..." << std::endl;

  const T dx = converter.getPhysDeltaX();
  auto domainI = makeSimulationDomain();
  auto fluidDomainI = makeFluidDomain(dx);

  // Keep a one-cell outer shell for boundary material 6, then recover the
  // interior fluid. OpenLB's discrete-normal detection needs this shell.
  sGeometry.rename(0, 6, domainI);
  sGeometry.rename(6, 1, fluidDomainI);

  Vector<T,3> inletOrigin{-20.0*D - dx/T(2), 0.0, -10.0*D + dx};
  Vector<T,3> inletExtent{dx, 3.0*D, 20.0*D - 2.0*dx};
  IndicatorCuboid3D<T> inletI(inletExtent, inletOrigin);
  sGeometry.rename(6, 3, inletI);

  Vector<T,3> outletOrigin{20.0*D - dx/T(2), 0.0, -10.0*D + dx};
  Vector<T,3> outletExtent{dx, 3.0*D, 20.0*D - 2.0*dx};
  IndicatorCuboid3D<T> outletI(outletExtent, outletOrigin);
  sGeometry.rename(6, 4, outletI);

  sGeometry.clean(true, std::vector<int>{1, 3, 4, 6});
  sGeometry.checkForErrors();
  sGeometry.print();
  printMaterialBounds(sGeometry, "level0");

  clout << "Prepare Geometry ... OK" << std::endl;
}

void prepareGeometryFine(const UnitConverter<T,DESCRIPTOR>& converter,
                         SuperGeometry<T,3>& sGeometry,
                         const std::string& name)
{
  OstreamManager clout(std::cout, "prepareGeometryFine");
  clout << "Prepare Geometry ..." << std::endl;

  sGeometry.rename(0, 1);
  auto cylinder = makeCylinder();
  sGeometry.rename(1, 2, *cylinder);

  sGeometry.clean(true, std::vector<int>{1, 2});
  //sGeometry.checkForErrors();
  sGeometry.print();
  printMaterialBounds(sGeometry, name);

  clout << "Prepare Geometry ... OK" << std::endl;
}

void prepareLattice(SuperLattice<T,DESCRIPTOR>& sLattice,
                    const UnitConverter<T,DESCRIPTOR>& converter,
                    SuperGeometry<T,3>& sGeometry,
                    T smagorinskyConstant)
{
  OstreamManager clout(std::cout, "prepareLattice");
  clout << "Prepare Lattice ..." << std::endl;

  const T omega = converter.getLatticeRelaxationFrequency();

  sLattice.defineDynamics<BulkDynamics>(sGeometry.getMaterialIndicator({1}));
  boundary::set<boundary::BounceBack>(sLattice, sGeometry, 2);

  using VelocityBC = boundary::InterpolatedVelocity<T, DESCRIPTOR, BulkDynamics>;
  using PressureBC = boundary::InterpolatedPressure<T, DESCRIPTOR, BulkDynamics>;
  boundary::set<VelocityBC>(sLattice, sGeometry, 3);
  boundary::set<PressureBC>(sLattice, sGeometry, 4);
  boundary::set<boundary::FullSlip>(sLattice, sGeometry, 6);

  AnalyticalConst3D<T,T> rhoF(T(1));
  AnalyticalConst3D<T,T> uZero(T(0), T(0), T(0));
  auto flowMaterials = sGeometry.getMaterialIndicator({1,3,4,6});
  auto solidMaterials = sGeometry.getMaterialIndicator({2});

  sLattice.defineRhoU(flowMaterials, rhoF, uZero);
  sLattice.iniEquilibrium(flowMaterials, rhoF, uZero);
  sLattice.defineRhoU(solidMaterials, rhoF, uZero);
  sLattice.iniEquilibrium(solidMaterials, rhoF, uZero);
  sLattice.setParameter<descriptors::OMEGA>(omega);
  sLattice.setParameter<collision::LES::SMAGORINSKY>(smagorinskyConstant);

  sLattice.initialize();
  clout << "Prepare Lattice ... OK" << std::endl;
}

void setBoundaryValues(SuperLattice<T,DESCRIPTOR>& sLattice,
                       const UnitConverter<T,DESCRIPTOR>& converter,
                       std::size_t iT,
                       SuperGeometry<T,3>& sGeometry)
{
  const std::size_t iTmaxStart = converter.getLatticeTime(rampTime);
  const std::size_t iTupdate = 30;

  if (iT % iTupdate == 0 && iT <= iTmaxStart) {
    PolynomialStartScale<T,std::size_t> startScale(iTmaxStart, T(1));
    std::size_t iTvec[1] = {iT};
    T frac[1] = {};
    startScale(frac, iTvec);

    const T uLattice = frac[0]*converter.getCharLatticeVelocity();
    AnalyticalConst3D<T,T> uInlet(uLattice, T(0), T(0));
    sLattice.defineU(sGeometry, 3, uInlet);

    sLattice.setProcessingContext<Array<momenta::FixedVelocityMomentumGeneric::VELOCITY>>(
      ProcessingContext::Simulation);
  }
}

void writeResults(SuperLattice<T,DESCRIPTOR>& sLattice,
                  const UnitConverter<T,DESCRIPTOR>& converter,
                  std::size_t iT,
                  SuperGeometry<T,3>& sGeometry,
                  const std::string& name)
{
  OstreamManager cloutVtk(std::cout, "vtkOutput");
  if (iT == 0) {
    SuperVTMwriter3D<T> vtmWriter(name, 0);
    if (name == "level0") {
      SuperLatticeRank3D rank(sLattice);
      vtmWriter.write(rank);
    }
    SuperGeometryF<T,DESCRIPTOR::d> geometryF(sGeometry);
    geometryF.getName() = name + "_geometry";
    vtmWriter.write(geometryF);
    vtmWriter.createMasterFile();
  }

  // Constructing a material indicator updates geometry statistics and issues
  // MPI collectives. Keep that on the main thread; only VTK file writing is
  // scheduled on the background pool.
  sLattice.setProcessingContext(ProcessingContext::Evaluation);
  std::shared_ptr<SuperIndicatorF3D<T>> outputMaterials(
    sGeometry.getMaterialIndicator({1,3,4,6}).release());

  sLattice.scheduleBackgroundOutputVTK(
    [&,name,iT,outputMaterials](auto task) {
      SuperVTMwriter3D<T> vtmWriter(name);
      SuperLatticePhysVelocity3D velocityF(sLattice, converter);
      SuperLatticePhysPressure3D pressureF(sLattice, converter);
      SuperLatticeRefinementMetricKnudsen3D qualityF(sLattice, converter);

      // Physical vorticity (curl of the velocity field), 1/s.
      std::list<int> fluidMats{1};
      SuperLatticePhysVorticityFD3D<T,DESCRIPTOR> vorticityF(
        sGeometry, sLattice, fluidMats, converter);
      SuperIdentityOnSuperIndicatorF3D<T,T> vorticityFluidF(
        vorticityF, *outputMaterials, T(0));
      vorticityFluidF.getName() = "physVorticity";

      // Dimensionless spanwise vorticity: omega_y * D / U0. The cylinder axis
      // is along y, so the y-component of the vorticity is the spanwise one.
      SuperExtractComponentF3D<T,T> omegaYF(vorticityF, 1);
      const T spanVortScale = D / converter.getCharPhysVelocity();
      SuperCalcMultiplication3D<T,T> omegaYNormF(spanVortScale, omegaYF);
      SuperIdentityOnSuperIndicatorF3D<T,T> omegaYFluidF(
        omegaYNormF, *outputMaterials, T(0));
      omegaYFluidF.getName() = "spanwiseVorticityD";

      SuperIdentityOnSuperIndicatorF3D<T,T> velocityFluidF(
        velocityF, *outputMaterials, T(0));
      SuperIdentityOnSuperIndicatorF3D<T,T> pressureFluidF(
        pressureF, *outputMaterials, T(0));
      velocityFluidF.getName() = velocityF.getName();
      pressureFluidF.getName() = pressureF.getName();
      vtmWriter.addFunctor(qualityF);
      vtmWriter.addFunctor(velocityFluidF);
      vtmWriter.addFunctor(pressureFluidF);
      vtmWriter.addFunctor(vorticityFluidF);
      vtmWriter.addFunctor(omegaYFluidF);
      task(vtmWriter, iT);
    });
  // Do not return to Simulation context while the GPU-block VTK tasks are
  // still reading Evaluation data. Without this wait, coarser levels could
  // write stale or corrupted fields while force sampling masks the race on
  // the finest level.
  sLattice.template waitForBackgroundTasks<stage::PreContextSwitchTo<ProcessingContext::Evaluation>>();
  cloutVtk << "Scheduled background VTK output for " << name
           << " at iT=" << iT << std::endl;
}

// ---------------------------------------------------------------------------
// Single-cylinder benchmark monitoring: forces, Strouhal number, surface Cp
// ---------------------------------------------------------------------------

const T kPi = std::acos(T(-1));
const std::size_t numCpAngles = 36;

struct ForceSample {
  T t;
  T fx;
  T fz;
};

struct CpAccumulator {
  std::vector<T> sum;
  std::vector<T> sumSq;
  std::size_t count;

  CpAccumulator(std::size_t n)
    : sum(n, T(0)), sumSq(n, T(0)), count(0) { }
};

// Time-averaged resolved 2D turbulent kinetic energy on the mid-span plane,
// following Brès et al. (2012), Sec. V.C:
//   TKE2D = 1/2 * ( u'u' + w'w' ) / U0^2
// where u', w' are the fluctuating in-plane velocity components (u = streamwise
// x, w = cross-stream z; the cylinder axis is the spanwise y, so the mid-span
// plane is y = 1.5D).  Velocity moments are accumulated on a bounded 2D grid
// and written to a standalone ImageData VTK file at the end.
struct TkePlaneSampler {
  std::vector<T> sumU, sumW, sumU2, sumW2;
  std::size_t nx, nz, count;
  T xMin, zMin, dx, dz, yPlane;

  TkePlaneSampler(T xm, T xM, T zm, T zM, T y, std::size_t nx_, std::size_t nz_)
    : nx(nx_), nz(nz_), count(0),
      xMin(xm), zMin(zm), yPlane(y),
      dx((xM - xm) / T(nx_)), dz((zM - zm) / T(nz_)),
      sumU(nx_*nz_, T(0)), sumW(nx_*nz_, T(0)),
      sumU2(nx_*nz_, T(0)), sumW2(nx_*nz_, T(0)) { }

  void accumulate(SuperLattice<T,DESCRIPTOR>& sLattice,
                  const UnitConverter<T,DESCRIPTOR>& converter)
  {
    sLattice.setProcessingContext(ProcessingContext::Evaluation);
    SuperLatticePhysVelocity3D<T,DESCRIPTOR> velF(sLattice, converter);
    AnalyticalFfromSuperF3D<T> interp(velF, true);

    std::size_t idx = 0;
    for (std::size_t j = 0; j < nz; ++j) {
      const T z = zMin + (T(j) + T(0.5)) * dz;
      for (std::size_t i = 0; i < nx; ++i) {
        const T x = xMin + (T(i) + T(0.5)) * dx;
        T pt[3] = {x, yPlane, z};
        T v[3] = {T(0), T(0), T(0)};
        interp(v, pt);
        sumU[idx] += v[0];
        sumU2[idx] += v[0] * v[0];
        sumW[idx] += v[2];
        sumW2[idx] += v[2] * v[2];
        ++idx;
      }
    }
    sLattice.setProcessingContext(ProcessingContext::Simulation);
    ++count;
  }

  void write(const std::string& path, T U0) const
  {
    std::ofstream fout(path.c_str(), std::ios::out);
    if (!fout.is_open()) {
      OstreamManager clout(std::cout, "tke2D");
      clout << "Could not write " << path << std::endl;
      return;
    }
    fout << "<?xml version=\"1.0\"?>\n";
    fout << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n";
    fout << "  <ImageData WholeExtent=\"0 " << (nx - 1)
         << " 0 0 0 " << (nz - 1)
         << "\" Origin=\"" << xMin << " " << (yPlane - 0.5*dx) << " " << zMin
         << "\" Spacing=\"" << dx << " " << dx << " " << dz << "\">\n";
    fout << "    <Piece Extent=\"0 " << (nx - 1) << " 0 0 0 " << (nz - 1) << "\">\n";
    fout << "      <PointData>\n";
    fout << "        <DataArray type=\"Float64\" Name=\"turbulentKineticEnergy2D\" "
         << "NumberOfComponents=\"1\" format=\"ascii\">\n";
    const T denom = U0 * U0;
    for (std::size_t k = 0; k < sumU.size(); ++k) {
      const T meanU = (count > 0) ? sumU[k] / T(count) : T(0);
      const T meanU2 = (count > 0) ? sumU2[k] / T(count) : T(0);
      const T varU = std::max(T(0), meanU2 - meanU * meanU);
      const T meanW = (count > 0) ? sumW[k] / T(count) : T(0);
      const T meanW2 = (count > 0) ? sumW2[k] / T(count) : T(0);
      const T varW = std::max(T(0), meanW2 - meanW * meanW);
      fout << (T(0.5) * (varU + varW) / denom) << "\n";
    }
    fout << "        </DataArray>\n";
    fout << "      </PointData>\n";
    fout << "    </Piece>\n";
    fout << "  </ImageData>\n";
    fout << "</VTKFile>\n";
  }
};

/// Read the physical force on a solid material and return (Fx, Fy, Fz) [N].
void readForce(SuperLattice<T,DESCRIPTOR>& sLattice,
               SuperGeometry<T,3>& sGeometry,
               int material,
               const UnitConverter<T,DESCRIPTOR>& converter,
               T force[3])
{
  SuperLatticePhysBoundaryForce3D<T,DESCRIPTOR> forceF(sLattice, sGeometry, material, converter);
  SuperSum3D<T,T> sumF(forceF, sGeometry, material);
  // SuperSum3D outputs (fx, fy, fz, voxelCount)
  T total[4] = {};
  int in[3] = {0, 0, 0};
  sumF(total, in);
  force[0] = total[0];
  force[1] = total[1];
  force[2] = total[2];
}

/// Record one force sample on the finest level.
void sampleForces(SuperLattice<T,DESCRIPTOR>& sLattice,
                  SuperGeometry<T,3>& sGeometry,
                  const UnitConverter<T,DESCRIPTOR>& converter,
                  T physTime,
                  std::vector<ForceSample>& samples)
{
  sLattice.setProcessingContext(ProcessingContext::Evaluation);

  T force[3] = {};
  readForce(sLattice, sGeometry, 2, converter, force);

  samples.push_back({physTime, force[0], force[2]});
  sLattice.setProcessingContext(ProcessingContext::Simulation);
}

/// Accumulate (mean, RMS) surface pressure coefficients on a cylinder.
void accumulateSurfaceCp(SuperLattice<T,DESCRIPTOR>& sLattice,
                         SuperGeometry<T,3>& sGeometry,
                         const UnitConverter<T,DESCRIPTOR>& converter,
                         T xCenter,
                         T probeRadius,
                         T qInf,
                         const T refPoint[3],
                         CpAccumulator& acc)
{
  SuperLatticePhysPressure3D<T,DESCRIPTOR> pressure(sLattice, converter);
  AnalyticalFfromSuperF3D<T> interp(pressure, true);

  T pRef = T(0);
  interp(&pRef, refPoint);

  const T yMid = 1.5*D;
  for (std::size_t k = 0; k < acc.sum.size(); ++k) {
    const T theta = 2.0 * kPi * T(k) / T(acc.sum.size());
    T pt[3] = {xCenter - probeRadius * std::cos(theta), yMid, probeRadius * std::sin(theta)};
    T p = T(0);
    interp(&p, pt);
    const T cp = (p - pRef) / qInf;
    acc.sum[k] += cp;
    acc.sumSq[k] += cp * cp;
  }
  acc.count++;
}

/// Dominant Strouhal number of the cylinder lift signal (coarse DFT).
T dominantStrouhal(const std::vector<ForceSample>& samples,
                   std::size_t startIdx,
                   T U0,
                   T diameter)
{
  if (samples.size() - startIdx < 8) {
    return T(0);
  }
  const T t0 = samples[startIdx].t;
  T bestSt = T(0);
  T bestAmp = T(-1);

  for (int iSt = 50; iSt <= 500; ++iSt) {
    const T St = T(iSt) / T(1000);
    const T omega = 2.0 * kPi * St * U0 / diameter;
    T re = T(0);
    T im = T(0);
    for (std::size_t k = startIdx; k < samples.size(); ++k) {
      const T dt = samples[k].t - t0;
      const T x = samples[k].fz;
      re += x * std::cos(omega * dt);
      im += x * std::sin(omega * dt);
    }
    const T amp = re * re + im * im;
    if (amp > bestAmp) {
      bestAmp = amp;
      bestSt = St;
    }
  }
  return bestSt;
}

void writeMonitors(const std::vector<ForceSample>& samples,
                   const CpAccumulator& cp,
                   int Re,
                   T U0,
                   T diameter,
                   T qInf,
                   T refArea,
                   T tTransient)
{
  if (!singleton::mpi().isMainProcessor()) {
    return;
  }

  OstreamManager clout(std::cout, "monitors");
  const std::string dir = singleton::directories().getLogOutDir();

  // Force time series.
  {
    std::ofstream fout((dir + "singleCylinder3d_Re3900_forces.dat").c_str(), std::ios::out);
    fout << std::scientific;
    fout << "# reference_Cd=1 reference_St=0.21\n";
    fout << "#t Cd Cl\n";
    for (const auto& s : samples) {
      fout << s.t
           << " " << s.fx/(qInf*refArea)
           << " " << s.fz/(qInf*refArea)
           << "\n";
    }
  }

  // Surface pressure coefficient distributions.
  {
    std::ofstream fout((dir + "singleCylinder3d_Re3900_cp.dat").c_str(), std::ios::out);
    fout << std::scientific;
    fout << "#theta_deg Cp_mean Cp_rms\n";
    for (std::size_t k = 0; k < cp.sum.size(); ++k) {
      const double thetaDeg = 360.0 * double(k) / double(cp.sum.size());
      const T mean = cp.count > 0 ? cp.sum[k]/T(cp.count) : T(0);
      const T ms = cp.count > 0 ? cp.sumSq[k]/T(cp.count) : T(0);
      const T rms = std::sqrt(std::max(T(0), ms - mean*mean));
      fout << thetaDeg << " " << mean << " " << rms << "\n";
    }
  }

  // Time-mean and RMS force coefficients + Strouhal number.
  {
    std::size_t startIdx = 0;
    for (std::size_t k = 0; k < samples.size(); ++k) {
      if (samples[k].t >= tTransient) {
        startIdx = k;
        break;
      }
    }
    const std::size_t n = samples.size() - startIdx;

    T meanCd = T(0), meanCl = T(0);
    T sqCd = T(0), sqCl = T(0);
    for (std::size_t k = startIdx; k < samples.size(); ++k) {
      const T cd = samples[k].fx/(qInf*refArea);
      const T cl = samples[k].fz/(qInf*refArea);
      meanCd += cd; sqCd += cd*cd;
      meanCl += cl; sqCl += cl*cl;
    }
    if (n > 0) {
      meanCd /= T(n); meanCl /= T(n);
      sqCd /= T(n); sqCl /= T(n);
    }
    const T rmsCd = std::sqrt(std::max(T(0), sqCd - meanCd*meanCd));
    const T rmsCl = std::sqrt(std::max(T(0), sqCl - meanCl*meanCl));

    const T St = dominantStrouhal(samples, startIdx, U0, diameter);
    const T freq = St * U0 / diameter;

    std::ofstream fout((dir + "singleCylinder3d_Re3900_stats.txt").c_str(), std::ios::out);
    fout << std::scientific;
    fout << "#quantity value\n";
    fout << "Re " << Re << "\n";
    fout << "U0 " << U0 << "\n";
    fout << "D " << diameter << "\n";
    fout << "Cd_target " << T(1.0) << "\n";
    fout << "St_target " << T(0.21) << "\n";
    fout << "mean_Cd " << meanCd << "\n";
    fout << "rms_Cd " << rmsCd << "\n";
    fout << "mean_Cl " << meanCl << "\n";
    fout << "rms_Cl " << rmsCl << "\n";
    fout << "St " << St << "\n";
    fout << "f_Hz " << freq << "\n";

    clout << "=== Single cylinder statistics (t >= " << tTransient << " s) ===" << std::endl;
    clout << "mean Cd:            " << meanCd << " (rms " << rmsCd << ", target 1.0)" << std::endl;
    clout << "mean Cl:            " << meanCl << " (rms " << rmsCl << ")" << std::endl;
    clout << "Strouhal St:        " << St << " (f = " << freq
          << " Hz, target 0.21)" << std::endl;
    clout << "Monitors written to: " << dir << std::endl;
  }
}

template <typename COARSE, typename FINE>
void addRefinementWeight(COARSE& coarse, FINE& fine, int factor)
{
  for (int iC=0; iC < coarse.size(); ++iC) {
    auto& cuboid = coarse.get(iC);
    cuboid.setWeight(cuboid.getLatticeVolume());
    const auto origin = cuboid.getOrigin();
    for (int jC=0; jC < fine.size(); ++jC) {
      if (fine.get(jC).getOrigin() == origin) {
        cuboid.setWeight(cuboid.getWeight() + factor*fine.get(jC).getWeight());
      }
    }
  }
}

int main(int argc, char* argv[])
{
  initialize(&argc, &argv);
  OstreamManager clout(std::cout, "main");

  CLIreader args(argc, argv);
  const int N = args.getValueOrFallback<int>("--resolution", 16);
  const T smagorinskyConstant = args.getValueOrFallback<T>("--smagorinsky", 0.15);
  const int Re = std::max(1, args.getValueOrFallback<int>("--reynolds", 3900));
  const T maxPhysT = args.getValueOrFallback<T>("--maxPhysT", T(12));
  const T outputInterval = args.getValueOrFallback<T>("--outputInterval", T(0.5));
  const T forceSampleDT = args.getValueOrFallback<T>("--forceSampleDT", T(0.005));
  const T cpSampleDT = args.getValueOrFallback<T>("--cpSampleDT", T(0.01));
  const T tkeSampleDT = args.getValueOrFallback<T>("--tkeSampleDT", T(2e-2));
  const T tTransient = args.getValueOrFallback<T>("--tTransient", T(7));

  const T physViscosity = inletVelocity * D / Re;
  // latticeU directly controls physDeltaT: dt = latticeU * D / (U_phys * N).
  // Higher latticeU → larger time step, but also larger Mach number.
  // Ma = latticeU * sqrt(3).  Typical ranges:
  //   latticeU=0.03 → Ma≈0.05 (conservative), 0.06 → Ma≈0.10,
  //   0.10 → Ma≈0.17 (max for engineering LES).
  // --mach overrides --latticeU; if neither is given, default latticeU=0.05.
  const T csLattice = T(1) / std::sqrt(T(3));  // lattice sound speed for D3Q19
  T charLatticeU;
  T machNumber;
  if (args.contains("--mach")) {
    machNumber = args.getValueOrFallback<T>("--mach", T(0.08));
    charLatticeU = machNumber * csLattice;
  } else {
    charLatticeU = args.getValueOrFallback<T>("--latticeU", T(0.05));
    machNumber = charLatticeU / csLattice;
  }
  if (!(machNumber < T(0.1))) {
    throw std::invalid_argument("Mach number must be below 0.1");
  }
  clout << "Mach number: " << machNumber << " (latticeU=" << charLatticeU << ")" << std::endl;
  const UnitConverterFromResolutionAndLatticeVelocity<T, DESCRIPTOR> converterLevel0(
    int {N},
    charLatticeU,
    (T) D,
    (T) inletVelocity,
    (T) physViscosity,
    (T) 1.225);
  converterLevel0.print();

  const T rho = converterLevel0.getPhysDensity();
  const T U0 = converterLevel0.getCharPhysVelocity();
  const T span = 3.0*D;
  const T refArea = D * span;                // projected frontal area = D * span
  const T qInf = T(0.5) * rho * U0 * U0;     // dynamic pressure
  const std::size_t forceSampleStep =
    std::max<std::size_t>(1, converterLevel0.getLatticeTime(forceSampleDT));
  const std::size_t cpSampleStep =
    std::max<std::size_t>(1, converterLevel0.getLatticeTime(cpSampleDT));
  const std::size_t tkeSampleStep =
    std::max<std::size_t>(1, converterLevel0.getLatticeTime(tkeSampleDT));

  if (N % 2 != 0) {
    throw std::runtime_error("The domain split requires resolution N to be divisible by 2");
  }

  auto simulationDomainI = makeSimulationDomain();
  CuboidDecomposition3D<T> cuboidDecompositionLevel0(
    simulationDomainI, converterLevel0.getPhysDeltaX());
  cuboidDecompositionLevel0.setPeriodicity({false, true, false});

  // Level0 outer partition: extend the Level1 core to the domain boundary.
  splitCuboidsAtPhysR(cuboidDecompositionLevel0, 0,
                      {-5.0*D, 15.0*D});
  splitCuboidsAtPhysR(cuboidDecompositionLevel0, 2,
                      {-5.0*D, 5.0*D});

  // Level1 ring around the Level2 core.
  splitCuboidsInCoreAtPhysR(
    cuboidDecompositionLevel0, 0, {-2.0*D, 11.0*D},
    Vector<T,3>{-5.0*D, 0.0, -5.0*D}, Vector<T,3>{20.0*D, 3.0*D, 10.0*D});
  splitCuboidsInCoreAtPhysR(
    cuboidDecompositionLevel0, 2, {-1.5*D, 1.5*D},
    Vector<T,3>{-5.0*D, 0.0, -5.0*D}, Vector<T,3>{20.0*D, 3.0*D, 10.0*D});

  // Level2 ring around the Level3 core.
  splitCuboidsInCoreAtPhysR(
    cuboidDecompositionLevel0, 0, {-1.0*D, 9.0*D},
    Vector<T,3>{-2.0*D, 0.0, -1.5*D}, Vector<T,3>{13.0*D, 3.0*D, 3.0*D});
  splitCuboidsInCoreAtPhysR(
    cuboidDecompositionLevel0, 2, {-1.0*D, 1.0*D},
    Vector<T,3>{-2.0*D, 0.0, -1.5*D}, Vector<T,3>{13.0*D, 3.0*D, 3.0*D});

  // Level3 core: split at z=0, then split each half into four equal parts
  // along x. The resulting eight blocks are each 2.5D by 1D in x/z.
  splitCuboidsInCoreAtPhysR(
    cuboidDecompositionLevel0, 0,
    {1.5*D, 4.0*D, 6.5*D},
    Vector<T,3>{-1.0*D, 0.0, -1.0*D}, Vector<T,3>{10.0*D, 3.0*D, 2.0*D});
  splitCuboidsInCoreAtPhysR(
    cuboidDecompositionLevel0, 2, {0.0*D},
    Vector<T,3>{-1.0*D, 0.0, -1.0*D}, Vector<T,3>{10.0*D, 3.0*D, 2.0*D});
  cuboidDecompositionLevel0.print();
  cuboidDecompositionLevel0.printExtended();
  printCuboidNodeBounds("level0", cuboidDecompositionLevel0);

  auto cuboidDecompositionLevel1 = cuboidDecompositionLevel0;
  const T dx0 = converterLevel0.getPhysDeltaX();
  const T dx1 = dx0 / T(2);
  const T dx2 = dx0 / T(4);
  const T level1SelectionMargin = T(3)*dx0;
  IndicatorCuboid3D<T> level1I(
    Vector<T,3>{20.0*D - 2.0*level1SelectionMargin,
                3.0*D,
                10.0*D - 2.0*level1SelectionMargin},
    Vector<T,3>{-5.0*D + level1SelectionMargin,
                0.0,
                -5.0*D + level1SelectionMargin});
  cuboidDecompositionLevel1.remove(level1I);
  cuboidDecompositionLevel1.refine(2);
  cuboidDecompositionLevel1.print();
  cuboidDecompositionLevel1.printExtended();
  printCuboidNodeBounds("level1", cuboidDecompositionLevel1);

  auto cuboidDecompositionLevel2 = cuboidDecompositionLevel1;
  const T level2SelectionMargin = T(3)*dx1;
  IndicatorCuboid3D<T> level2I(
    Vector<T,3>{13.0*D - 2.0*level2SelectionMargin,
                3.0*D,
                3.0*D - 2.0*level2SelectionMargin},
    Vector<T,3>{-2.0*D + level2SelectionMargin,
                0.0,
                -1.5*D + level2SelectionMargin});
  cuboidDecompositionLevel2.remove(level2I);
  cuboidDecompositionLevel2.refine(2);
  cuboidDecompositionLevel2.print();
  cuboidDecompositionLevel2.printExtended();
  printCuboidNodeBounds("level2", cuboidDecompositionLevel2);

  auto cuboidDecompositionLevel3 = cuboidDecompositionLevel2;

  const T level3SelectionMargin = T(3)*dx2;
  IndicatorCuboid3D<T> level3I(
    Vector<T,3>{10.0*D - 2.0*level3SelectionMargin,
                3.0*D,
                2.0*D - 2.0*level3SelectionMargin},
    Vector<T,3>{-1.0*D + level3SelectionMargin,
                0.0,
                -1.0*D + level3SelectionMargin});

  cuboidDecompositionLevel3.remove(level3I);
  cuboidDecompositionLevel3.refine(2);
  cuboidDecompositionLevel3.print();
  cuboidDecompositionLevel3.printExtended();
  printCuboidNodeBounds("level3", cuboidDecompositionLevel3);

  addRefinementWeight(cuboidDecompositionLevel2, cuboidDecompositionLevel3, 2);
  addRefinementWeight(cuboidDecompositionLevel1, cuboidDecompositionLevel2, 2);
  addRefinementWeight(cuboidDecompositionLevel0, cuboidDecompositionLevel1, 2);

  HeuristicLoadBalancer<T> loadBalancerLevel0(cuboidDecompositionLevel0);
  SuperGeometry<T,3> sGeometryLevel0(cuboidDecompositionLevel0, loadBalancerLevel0, 3);
  prepareGeometry(converterLevel0, sGeometryLevel0);
  SuperLattice<T,DESCRIPTOR> sLatticeLevel0(cuboidDecompositionLevel0,
                                            loadBalancerLevel0,
                                            3,
                                            converterLevel0);
  prepareLattice(sLatticeLevel0, converterLevel0, sGeometryLevel0,
                  smagorinskyConstant);

  RefinedLoadBalancer<T,3> loadBalancerLevel1(cuboidDecompositionLevel0,
                                              loadBalancerLevel0,
                                              cuboidDecompositionLevel1);
  auto converterLevel1 = convectivelyRefineUnitConverter(converterLevel0, 2);
  converterLevel1.print();
  SuperGeometry<T,3> sGeometryLevel1(cuboidDecompositionLevel1, loadBalancerLevel1);
  prepareGeometryFine(converterLevel1, sGeometryLevel1, "level1");
  SuperLattice<T,DESCRIPTOR> sLatticeLevel1(cuboidDecompositionLevel1,
                                            loadBalancerLevel1,
                                            3,
                                            converterLevel1);
  prepareLattice(sLatticeLevel1, converterLevel1, sGeometryLevel1,
                  smagorinskyConstant);

  RefinedLoadBalancer<T,3> loadBalancerLevel2(cuboidDecompositionLevel1,
                                              loadBalancerLevel1,
                                              cuboidDecompositionLevel2);
  auto converterLevel2 = convectivelyRefineUnitConverter(converterLevel1, 2);
  converterLevel2.print();
  SuperGeometry<T,3> sGeometryLevel2(cuboidDecompositionLevel2, loadBalancerLevel2);
  prepareGeometryFine(converterLevel2, sGeometryLevel2, "level2");
  SuperLattice<T,DESCRIPTOR> sLatticeLevel2(cuboidDecompositionLevel2,
                                            loadBalancerLevel2,
                                            3,
                                            converterLevel2);
  prepareLattice(sLatticeLevel2, converterLevel2, sGeometryLevel2,
                  smagorinskyConstant);

  RefinedLoadBalancer<T,3> loadBalancerLevel3(cuboidDecompositionLevel2,
                                              loadBalancerLevel2,
                                              cuboidDecompositionLevel3);
  auto converterLevel3 = convectivelyRefineUnitConverter(converterLevel2, 2);
  converterLevel3.print();
  SuperGeometry<T,3> sGeometryLevel3(cuboidDecompositionLevel3, loadBalancerLevel3);
  prepareGeometryFine(converterLevel3, sGeometryLevel3, "level3");
  SuperLattice<T,DESCRIPTOR> sLatticeLevel3(cuboidDecompositionLevel3,
                                            loadBalancerLevel3,
                                            3,
                                            converterLevel3);
  prepareLattice(sLatticeLevel3, converterLevel3, sGeometryLevel3,
                  smagorinskyConstant);

  auto coarseToFine01 = refinement::lagrava::makeCoarseToFineCoupler(
    sLatticeLevel0, sGeometryLevel0, sLatticeLevel1, sGeometryLevel1);
  auto fineToCoarse01 = refinement::lagrava::makeFineToCoarseCoupler(
    sLatticeLevel0, sGeometryLevel0, sLatticeLevel1, sGeometryLevel1);
  auto coarseToFine12 = refinement::lagrava::makeCoarseToFineCoupler(
    sLatticeLevel1, sGeometryLevel1, sLatticeLevel2, sGeometryLevel2);
  auto fineToCoarse12 = refinement::lagrava::makeFineToCoarseCoupler(
    sLatticeLevel1, sGeometryLevel1, sLatticeLevel2, sGeometryLevel2);
  auto coarseToFine23 = refinement::lagrava::makeCoarseToFineCoupler(
    sLatticeLevel2, sGeometryLevel2, sLatticeLevel3, sGeometryLevel3);
  auto fineToCoarse23 = refinement::lagrava::makeFineToCoarseCoupler(
    sLatticeLevel2, sGeometryLevel2, sLatticeLevel3, sGeometryLevel3);

  coarseToFine01->apply(meta::id<refinement::lagrava::InitializeO>{});
  coarseToFine12->apply(meta::id<refinement::lagrava::InitializeO>{});
  coarseToFine23->apply(meta::id<refinement::lagrava::InitializeO>{});

  // Transfer the initial CPU-side lattice state to the GPU once, before
  // the main loop. Official OpenLB examples do not call
  // setProcessingContext(Simulation) inside the time loop; doing so copies
  // stale CPU data over the GPU-side collideAndStream results.
  sLatticeLevel0.setProcessingContext(ProcessingContext::Simulation);
  sLatticeLevel1.setProcessingContext(ProcessingContext::Simulation);
  sLatticeLevel2.setProcessingContext(ProcessingContext::Simulation);
  sLatticeLevel3.setProcessingContext(ProcessingContext::Simulation);

  // --- Benchmark monitoring state -----------------------------------------
  std::vector<ForceSample> forceSamples;
  CpAccumulator cp(numCpAngles);
  const T refPoint[3] = {-12.0*D, 1.5*D, 0.0};
  const T probeOffset = 1.5 * converterLevel3.getPhysDeltaX();
  const T probeRadius = D/2.0 + probeOffset;

  // 2D resolved TKE sampling on the mid-span plane (paper Sec. V.C).
  const std::size_t tkeNX = 200;
  const std::size_t tkeNZ = 80;
  TkePlaneSampler tkeSampler(-1.0*D, 8.0*D, -2.0*D, 2.0*D, 1.5*D, tkeNX, tkeNZ);

  auto writeSimulationInfo = [&]() {
    OstreamManager cloutInfo(std::cout, "simulationInfo");
    std::ostringstream os;
    os << "===== Single cylinder simulation parameters =====\n";
    os << "D                        = " << D << " m\n";
    os << "span (periodic)          = 3D = " << span << " m\n";
    os << "U0                       = " << U0 << " m/s\n";
    os << "Re                       = " << Re << "\n";
    os << "Mach                     = " << machNumber << "\n";
    os << "physViscosity            = " << physViscosity << " m^2/s\n";
    os << "rho                      = " << rho << " kg/m^3\n";
    os << "Smagorinsky constant     = " << smagorinskyConstant << "\n";
    os << "resolution N             = " << N << "\n";
    os << "latticeU                 = " << charLatticeU << "\n";
    os << "base dx                  = " << converterLevel0.getPhysDeltaX() << " m\n";
    os << "base dt                  = " << converterLevel0.getPhysDeltaT() << " s\n";
    os << "total physical time      = " << maxPhysT << " s\n";
    os << "total base steps         = " << converterLevel0.getLatticeTime(maxPhysT) << "\n";
    os << "ramp time (to U0)        = " << rampTime << " s\n";
    os << "forceSampleDT            = " << forceSampleDT << " s\n";
    os << "cpSampleDT               = " << cpSampleDT << " s\n";
    os << "tkeSampleDT              = " << tkeSampleDT << " s\n";
    os << "tTransient               = " << tTransient << " s\n";
    os << "Cd target                = 1.0\n";
    os << "St target                = 0.21\n";
    os << "VTK interval             = " << outputInterval << " s\n";
    os << "level0 cells             = " << sGeometryLevel0.getStatistics().getNvoxel() << "\n";
    os << "level1 cells             = " << sGeometryLevel1.getStatistics().getNvoxel() << "\n";
    os << "level2 cells             = " << sGeometryLevel2.getStatistics().getNvoxel() << "\n";
    os << "level3 cells             = " << sGeometryLevel3.getStatistics().getNvoxel() << "\n";
    os << "weighted total cells     = "
       << (sGeometryLevel0.getStatistics().getNvoxel()
           + 2*sGeometryLevel1.getStatistics().getNvoxel()
           + 4*sGeometryLevel2.getStatistics().getNvoxel()
           + 8*sGeometryLevel3.getStatistics().getNvoxel()) << "\n";
    os << "=================================================\n";
    cloutInfo << os.str();
    if (singleton::mpi().isMainProcessor()) {
      std::ofstream fout((singleton::directories().getLogOutDir()
                         + "singleCylinder3d_Re3900_simulationInfo.txt").c_str());
      if (fout.is_open()) {
        fout << os.str();
      }
    }
  };
  writeSimulationInfo();

  clout << "starting simulation..." << std::endl;
  const std::size_t iTmax = converterLevel0.getLatticeTime(maxPhysT);
  util::Timer<T> timer(iTmax,
                       sGeometryLevel0.getStatistics().getNvoxel()
                       + 2*sGeometryLevel1.getStatistics().getNvoxel()
                       + 4*sGeometryLevel2.getStatistics().getNvoxel()
                       + 8*sGeometryLevel3.getStatistics().getNvoxel());
  timer.start();

  // Define subcycling lambdas outside the loop (no need to recreate each iteration).
  auto advanceLevel3 = [&]() {
    sLatticeLevel3.collideAndStream();
    coarseToFine23->apply(meta::id<refinement::lagrava::HalfTimeCoarseToFineO>{});

    sLatticeLevel3.collideAndStream();
    coarseToFine23->apply(meta::id<refinement::lagrava::FullTimeCoarseToFineO>{});

    fineToCoarse23->apply(meta::id<refinement::lagrava::FineToCoarseO>{});
  };

  auto advanceLevel2 = [&]() {
    sLatticeLevel2.collideAndStream();
    coarseToFine12->apply(meta::id<refinement::lagrava::HalfTimeCoarseToFineO>{});
    advanceLevel3();

    sLatticeLevel2.collideAndStream();
    coarseToFine12->apply(meta::id<refinement::lagrava::FullTimeCoarseToFineO>{});
    advanceLevel3();

    fineToCoarse12->apply(meta::id<refinement::lagrava::FineToCoarseO>{});
  };

  for (std::size_t iT = 0; iT < iTmax; ++iT) {
    if (iT % converterLevel0.getLatticeTime(outputInterval) == 0) {
      writeResults(sLatticeLevel0, converterLevel0, iT, sGeometryLevel0, "level0");
      writeResults(sLatticeLevel1, converterLevel1, iT, sGeometryLevel1, "level1");
      writeResults(sLatticeLevel2, converterLevel2, iT, sGeometryLevel2, "level2");
      writeResults(sLatticeLevel3, converterLevel3, iT, sGeometryLevel3, "level3");
    }

    // Benchmark monitoring: forces on the finest level + surface Cp.
    if (iT % forceSampleStep == 0) {
      sampleForces(sLatticeLevel3, sGeometryLevel3, converterLevel3,
                   converterLevel0.getPhysTime(iT), forceSamples);
    }
    if (iT % cpSampleStep == 0 &&
        converterLevel0.getPhysTime(iT) >= tTransient) {
      sLatticeLevel3.setProcessingContext(ProcessingContext::Evaluation);
      accumulateSurfaceCp(sLatticeLevel3, sGeometryLevel3, converterLevel3,
                          0.0, probeRadius, qInf, refPoint, cp);
      sLatticeLevel3.setProcessingContext(ProcessingContext::Simulation);
    }
    if (iT % tkeSampleStep == 0 &&
        converterLevel0.getPhysTime(iT) >= tTransient) {
      tkeSampler.accumulate(sLatticeLevel3, converterLevel3);
    }

    setBoundaryValues(sLatticeLevel0, converterLevel0, iT, sGeometryLevel0);

    sLatticeLevel0.collideAndStream();

    sLatticeLevel1.collideAndStream();
    coarseToFine01->apply(meta::id<refinement::lagrava::HalfTimeCoarseToFineO>{});
    advanceLevel2();

    sLatticeLevel1.collideAndStream();
    coarseToFine01->apply(meta::id<refinement::lagrava::FullTimeCoarseToFineO>{});
    advanceLevel2();

    fineToCoarse01->apply(meta::id<refinement::lagrava::FineToCoarseO>{});

    // Print progress every 100 steps for responsive monitoring.
    if (iT % 100 == 0) {
      timer.update(iT);
      timer.printStep();
      sLatticeLevel0.getStatistics().print(iT, converterLevel0.getPhysTime(iT));
    }
  }

  singleton::pool().wait();

  writeMonitors(forceSamples, cp, Re, U0, D, qInf, refArea, tTransient);
  if (singleton::mpi().isMainProcessor()) {
    tkeSampler.write(singleton::directories().getLogOutDir()
                     + "singleCylinder3d_Re3900_tke2d.vti", U0);
  }

  timer.stop();

  timer.printSummary();
  if (singleton::mpi().isMainProcessor()) {
    std::ofstream timerSummaryFile((singleton::directories().getLogOutDir()
                                   + "singleCylinder3d_Re3900_timerSummary.txt").c_str());
    if (timerSummaryFile.is_open()) {
      OstreamManager timerSummaryOut(timerSummaryFile, "timerSummary");
      timer.printSummary(timerSummaryOut);
    }
  }
}
