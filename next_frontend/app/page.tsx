"use client";

import { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

const ELEMENTS = [
  "Ag", "Al", "B", "C", "Ca", "Co", "Cr", "Cu", "Fe",
  "Ga", "Hf", "I", "Li", "Mg", "Mn", "Mo", "Nb", "Nd",
  "Ni", "O", "Pd", "Re", "Ru", "S", "Sc", "Si", "Sn",
  "T", "Ta", "Ti", "V", "W", "Y", "Zn", "Zr"
];

const initialComposition: Record<string, string> = {};

ELEMENTS.forEach((element) => {
  initialComposition[element] = "";
});

export default function Home() {
  const [composition, setComposition] =
    useState<Record<string, string>>(initialComposition);

  const [temperature, setTemperature] = useState("25");
  const [phase, setPhase] = useState("FCC");
  const [testType, setTestType] = useState("TENSILE");

  const [grainSize, setGrainSize] = useState("");
  const [densityExp, setDensityExp] = useState("");
  const [densityCalc, setDensityCalc] = useState("");
  const [precipitateSize, setPrecipitateSize] = useState("");
  const [matrixVolume, setMatrixVolume] = useState("");
  const [youngsModulusExp, setYoungsModulusExp] = useState("");
  const [youngsModulusCalc, setYoungsModulusCalc] = useState("");

  const [processingMethod, setProcessingMethod] = useState("");
  const [alloyClass, setAlloyClass] = useState("");
  const [equilibriumCondition, setEquilibriumCondition] = useState("");
  const [singleMultiphase, setSingleMultiphase] = useState("");
  const [precipitateInfo, setPrecipitateInfo] = useState("");

  const [prediction, setPrediction] = useState<{
    base_prediction_mpa: number;
    residual_correction_mpa: number;
    yield_strength_prediction_mpa: number;
  } | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateElement = (element: string, value: string) => {
    setComposition((current) => ({
      ...current,
      [element]: value,
    }));
  };

  const resetForm = () => {
    setComposition({ ...initialComposition });
    setTemperature("25");
    setPhase("FCC");
    setTestType("TENSILE");

    setGrainSize("");
    setDensityExp("");
    setDensityCalc("");
    setPrecipitateSize("");
    setMatrixVolume("");
    setYoungsModulusExp("");
    setYoungsModulusCalc("");

    setProcessingMethod("");
    setAlloyClass("");
    setEquilibriumCondition("");
    setSingleMultiphase("");
    setPrecipitateInfo("");

    setPrediction(null);
    setError("");
  };

  const loadCantorPreset = () => {
    const preset = { ...initialComposition };

    ["Co", "Cr", "Fe", "Ni", "Mn"].forEach((element) => {
      preset[element] = "20";
    });

    setComposition(preset);
    setPrediction(null);
    setError("");
  };

  const predictYieldStrength = async () => {
    setLoading(true);
    setError("");
    setPrediction(null);

    try {
      const compositionPayload: Record<string, number> = {};

      ELEMENTS.forEach((element) => {
        const value = composition[element];

        if (value !== "") {
          const numericValue = Number(value);

          if (!Number.isFinite(numericValue) || numericValue < 0) {
            throw new Error(`Invalid composition value for ${element}.`);
          }

          compositionPayload[element] = numericValue;
        }
      });

      const totalComposition = Object.values(compositionPayload)
        .reduce((sum, value) => sum + value, 0);

      if (totalComposition <= 0) {
        throw new Error("Enter at least one elemental composition.");
      }

      if (Math.abs(totalComposition - 100) > 0.1) {
        throw new Error(
          `Total composition is ${totalComposition.toFixed(2)} at.%. Adjust the composition to 100 at.% before prediction.`
        );
      }

      const response = await fetch(`${API_URL}/predict-material`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          composition: compositionPayload,
          Test_Temperature_C: Number(temperature),
          Grain_Size_um: grainSize === "" ? null : Number(grainSize),
          Density_Exp_g_cm3: densityExp === "" ? null : Number(densityExp),
          Density_Calc_g_cm3: densityCalc === "" ? null : Number(densityCalc),
          Precipitate_Size_nm: precipitateSize === "" ? null : Number(precipitateSize),
          Matrix_Volume_pct: matrixVolume === "" ? null : Number(matrixVolume),
          Youngs_Modulus_Exp_GPa: youngsModulusExp === "" ? null : Number(youngsModulusExp),
          Youngs_Modulus_Calc_GPa: youngsModulusCalc === "" ? null : Number(youngsModulusCalc),

          Processing_Method: processingMethod || null,
          Phase: phase,
          Alloy_Class: alloyClass || null,
          Equilibrium_Condition: equilibriumCondition || null,
          Single_Multiphase: singleMultiphase || null,
          Test_Type: testType,
          Precipitate_Info: precipitateInfo || null,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction failed.");
      }

      setPrediction(data);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Unable to connect to the prediction API."
      );
    } finally {
      setLoading(false);
    }
  };

  const totalComposition = Object.values(composition)
    .reduce((sum, value) => sum + (value === "" ? 0 : Number(value)), 0);

  return (
    <main className="min-h-screen bg-slate-950 text-white">
      <div className="mx-auto max-w-7xl px-6 py-10 lg:px-10">

        <header className="mb-10">
          <div className="mb-3 inline-flex rounded-full border border-cyan-400/30 bg-cyan-400/10 px-4 py-1 text-sm text-cyan-300">
            AI-Assisted Materials Discovery
          </div>

          <h1 className="text-4xl font-bold tracking-tight md:text-5xl">
            HEA / MPEA Yield Strength Predictor
          </h1>

          <p className="mt-4 max-w-3xl text-lg text-slate-400">
            Predict tensile yield strength using composition, materials
            informatics descriptors, phase-aware feature engineering,
            XGBoost regression, specialist models, and residual correction.
          </p>
        </header>

        <div className="grid gap-8 lg:grid-cols-[1.5fr_1fr]">

          <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl">

            <div className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <h2 className="text-xl font-semibold">
                  Alloy Composition
                </h2>
                <p className="mt-1 text-sm text-slate-400">
                  Enter elemental composition in atomic percent.
                </p>
              </div>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={loadCantorPreset}
                  className="rounded-lg border border-cyan-400/30 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-300 transition hover:bg-cyan-400/20"
                >
                  Cantor HEA
                </button>

                <button
                  type="button"
                  onClick={resetForm}
                  className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition hover:bg-slate-800"
                >
                  Reset
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
              {ELEMENTS.map((element) => (
                <div key={element}>
                  <label className="mb-1 block text-xs font-medium text-slate-400">
                    {element}
                  </label>

                  <input
                    type="number"
                    min="0"
                    max="100"
                    step="0.01"
                    value={composition[element]}
                    onChange={(event) =>
                      updateElement(element, event.target.value)
                    }
                    placeholder="0"
                    className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-400"
                  />
                </div>
              ))}
            </div>

            <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950 p-4">
              <div className="mb-2 flex items-center justify-between">
                <span className="text-sm text-slate-400">
                  Composition completeness
                </span>

                <span className={`text-sm font-semibold ${
                  Math.abs(totalComposition - 100) <= 0.1
                    ? "text-emerald-400"
                    : "text-amber-400"
                }`}>
                  {totalComposition.toFixed(2)} / 100 at.%
                </span>
              </div>

              <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                <div
                  className={`h-full rounded-full transition-all ${
                    totalComposition > 100
                      ? "bg-red-400"
                      : "bg-cyan-400"
                  }`}
                  style={{
                    width: `${Math.min(totalComposition, 100)}%`
                  }}
                />
              </div>

              {totalComposition > 0 && (
                <div className="mt-4 space-y-2">
                  {ELEMENTS
                    .filter((element) => Number(composition[element] || 0) > 0)
                    .sort(
                      (a, b) =>
                        Number(composition[b] || 0) -
                        Number(composition[a] || 0)
                    )
                    .slice(0, 5)
                    .map((element) => {
                      const value = Number(composition[element] || 0);

                      return (
                        <div key={element}>
                          <div className="mb-1 flex justify-between text-xs">
                            <span className="text-slate-400">{element}</span>
                            <span className="text-slate-300">
                              {value.toFixed(2)}%
                            </span>
                          </div>

                          <div className="h-1.5 overflow-hidden rounded-full bg-slate-800">
                            <div
                              className="h-full rounded-full bg-cyan-400"
                              style={{
                                width: `${Math.min(value, 100)}%`
                              }}
                            />
                          </div>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-900/80 p-6 shadow-xl">

            <h2 className="text-xl font-semibold">
              Material Conditions
            </h2>

            <p className="mt-1 text-sm text-slate-400">
              Provide experimental conditions used for prediction.
            </p>

            <div className="mt-6 space-y-5">

              <div>
                <label className="mb-2 block text-sm text-slate-300">
                  Test Temperature (°C)
                </label>

                <input
                  type="number"
                  value={temperature}
                  onChange={(event) => setTemperature(event.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                />
              </div>

              <div>
                <label className="mb-2 block text-sm text-slate-300">
                  Phase
                </label>

                <select
                  value={phase}
                  onChange={(event) => setPhase(event.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                >
                  <option value="FCC">FCC</option>
                  <option value="BCC">BCC</option>
                  <option value="HCP">HCP</option>
                  <option value="B2">B2</option>
                  <option value="LAVES">Laves</option>
                  <option value="SIGMA">Sigma</option>
                  <option value="L12">L12</option>
                  <option value="COMPLEX">Complex</option>
                </select>
              </div>

              <div>
                <label className="mb-2 block text-sm text-slate-300">
                  Test Type
                </label>

                <select
                  value={testType}
                  onChange={(event) => setTestType(event.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                >
                  <option value="TENSILE">Tensile</option>
                  <option value="COMPRESSION">Compression</option>
                </select>
              </div>

              <div className="mt-7 border-t border-slate-800 pt-6">
                <h3 className="text-lg font-semibold text-white">
                  Advanced Material Properties
                </h3>

                <p className="mt-1 text-sm text-slate-400">
                  Optional experimental values. Leave blank when unavailable.
                </p>

                <div className="mt-5 grid gap-4 sm:grid-cols-2">

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Grain Size (?m)
                    </label>
                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      value={grainSize}
                      onChange={(event) => setGrainSize(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Experimental Density (g/cm?)
                    </label>
                    <input
                      type="number"
                      min="0"
                      step="0.001"
                      value={densityExp}
                      onChange={(event) => setDensityExp(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Calculated Density (g/cm?)
                    </label>
                    <input
                      type="number"
                      min="0"
                      step="0.001"
                      value={densityCalc}
                      onChange={(event) => setDensityCalc(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Precipitate Size (nm)
                    </label>
                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      value={precipitateSize}
                      onChange={(event) => setPrecipitateSize(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Matrix Volume (%)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      step="0.01"
                      value={matrixVolume}
                      onChange={(event) => setMatrixVolume(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Experimental Young&apos;s Modulus (GPa)
                    </label>
                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      value={youngsModulusExp}
                      onChange={(event) => setYoungsModulusExp(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Calculated Young&apos;s Modulus (GPa)
                    </label>
                    <input
                      type="number"
                      min="0"
                      step="0.01"
                      value={youngsModulusCalc}
                      onChange={(event) => setYoungsModulusCalc(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                </div>
              </div>

              <div className="mt-7 border-t border-slate-800 pt-6">
                <h3 className="text-lg font-semibold text-white">
                  Processing Information
                </h3>

                <p className="mt-1 text-sm text-slate-400">
                  Optional categorical information from the material record.
                </p>

                <div className="mt-5 space-y-4">

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Processing Method
                    </label>
                    <input
                      type="text"
                      value={processingMethod}
                      onChange={(event) => setProcessingMethod(event.target.value)}
                      placeholder="e.g. Casting, Annealing, Additive Manufacturing"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Alloy Class
                    </label>
                    <input
                      type="text"
                      value={alloyClass}
                      onChange={(event) => setAlloyClass(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">

                    <div>
                      <label className="mb-2 block text-sm text-slate-300">
                        Equilibrium Condition
                      </label>
                      <input
                        type="text"
                        value={equilibriumCondition}
                        onChange={(event) => setEquilibriumCondition(event.target.value)}
                        placeholder="Optional"
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                      />
                    </div>

                    <div>
                      <label className="mb-2 block text-sm text-slate-300">
                        Single / Multiphase
                      </label>
                      <select
                        value={singleMultiphase}
                        onChange={(event) => setSingleMultiphase(event.target.value)}
                        className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                      >
                        <option value="">Not specified</option>
                        <option value="Single">Single phase</option>
                        <option value="Multiphase">Multiphase</option>
                      </select>
                    </div>

                  </div>

                  <div>
                    <label className="mb-2 block text-sm text-slate-300">
                      Precipitate Information
                    </label>
                    <input
                      type="text"
                      value={precipitateInfo}
                      onChange={(event) => setPrecipitateInfo(event.target.value)}
                      placeholder="Optional"
                      className="w-full rounded-lg border border-slate-700 bg-slate-950 px-4 py-3 text-white outline-none focus:border-cyan-400"
                    />
                  </div>

                </div>
              </div>

              <button
                type="button"
                onClick={predictYieldStrength}
                disabled={loading}
                className="w-full rounded-xl bg-cyan-400 px-5 py-4 font-semibold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? "Predicting..." : "Predict Yield Strength"}
              </button>

              {error && (
                <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
                  {error}
                </div>
              )}
            </div>
          </section>
        </div>

        {prediction && (
          <section className="mt-8 rounded-2xl border border-cyan-400/20 bg-slate-900/80 p-6 shadow-xl">

            <h2 className="text-xl font-semibold">
              Prediction Result
            </h2>

            <div className="mt-6 grid gap-4 md:grid-cols-3">

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-5">
                <p className="text-sm text-slate-400">
                  Base Model Prediction
                </p>
                <p className="mt-2 text-3xl font-bold">
                  {prediction.base_prediction_mpa.toFixed(2)}
                  <span className="ml-2 text-sm font-normal text-slate-500">
                    MPa
                  </span>
                </p>
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-5">
                <p className="text-sm text-slate-400">
                  Residual Correction
                </p>
                <p className="mt-2 text-3xl font-bold">
                  {prediction.residual_correction_mpa >= 0 ? "+" : ""}
                  {prediction.residual_correction_mpa.toFixed(2)}
                  <span className="ml-2 text-sm font-normal text-slate-500">
                    MPa
                  </span>
                </p>
              </div>

              <div className="rounded-xl border border-cyan-400/30 bg-cyan-400/10 p-5">
                <p className="text-sm text-cyan-300">
                  Predicted Yield Strength
                </p>
                <p className="mt-2 text-4xl font-bold text-cyan-300">
                  {prediction.yield_strength_prediction_mpa.toFixed(2)}
                  <span className="ml-2 text-sm font-normal text-cyan-400">
                    MPa
                  </span>
                </p>
              </div>

            </div>
          </section>
        )}

        <footer className="mt-10 text-center text-sm text-slate-600">
          1,941 experimental records · 359 production features ·
          XGBoost-based ensemble · FastAPI inference
        </footer>

      </div>
    </main>
  );
}
