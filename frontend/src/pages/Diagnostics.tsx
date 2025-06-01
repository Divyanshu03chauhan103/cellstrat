// src/pages/Diagnostics.tsx

import { FC, useState, useEffect, useRef } from "react";
import MainLayout from "../components/layout/MainLayout";

interface DiagnosticResult {
  severity: "normal" | "attention" | "critical";
  findings: string[];
  recommendations: string[];
  followUp?: string;
}

const DiagnosticsPage: FC = () => {
  const [activeTab, setActiveTab] = useState<"upload" | "results" | "history">("upload");
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<DiagnosticResult | null>(null);
  const [symptomInput, setSymptomInput] = useState("");
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [medicalHistory, setMedicalHistory] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [history, setHistory] = useState<DiagnosticResult[]>([]);

  useEffect(() => {
    const saved = localStorage.getItem("diagnosticHistory");
    if (saved) {
      setHistory(JSON.parse(saved));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem("diagnosticHistory", JSON.stringify(history));
  }, [history]);

  const commonSymptoms = [
    "Headache",
    "Fever",
    "Cough",
    "Fatigue",
    "Shortness of breath",
    "Chest pain",
    "Nausea",
    "Dizziness",
    "Back pain",
    "Joint pain",
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      setFiles((prev) => [...prev, ...newFiles]);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer.files) {
      const newFiles = Array.from(e.dataTransfer.files);
      setFiles((prev) => [...prev, ...newFiles]);
    }
  };

  const removeFile = (index: number) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleSymptomAdd = () => {
    const trimmed = symptomInput.trim();
    if (trimmed && !symptoms.includes(trimmed)) {
      setSymptoms([...symptoms, trimmed]);
      setSymptomInput("");
    }
  };

  const removeSymptom = (symptom: string) => {
    setSymptoms(symptoms.filter((s) => s !== symptom));
  };

  const addCommonSymptom = (symptom: string) => {
    if (!symptoms.includes(symptom)) {
      setSymptoms([...symptoms, symptom]);
    }
  };

  const saveToHistory = () => {
    if (results) {
      setHistory((prev) => [...prev, results]);
      alert("Result saved to history.");
    } else {
      alert("No result to save.");
    }
  };

  const analyzeDiagnostics = async () => {
    if (files.length === 0 && symptoms.length === 0 && medicalHistory.trim() === "") {
      alert("Please upload at least one medical report, add symptoms, or enter medical history.");
      return;
    }

    setIsAnalyzing(true);
    setActiveTab("results");

    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));
      formData.append("symptoms", JSON.stringify(symptoms));
      formData.append("medical_history", medicalHistory);

      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to analyze diagnostics");
      }

      const data: DiagnosticResult = await response.json();
      setResults(data);

      setHistory((prev) => {
        const updated = [...prev, data];
        localStorage.setItem("diagnosticHistory", JSON.stringify(updated));
        return updated;
      });
    } catch (error) {
      alert("Something went wrong while analyzing.");
      console.error("Analysis error:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderUploadTab = () => (
    <div className="p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Upload Medical Reports</h2>

      {/* File Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center mb-6 ${
          isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-gray-50"
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="flex flex-col items-center">
          <div className="w-16 h-16 mb-4 bg-blue-100 rounded-full flex items-center justify-center">
            <svg
              className="w-8 h-8 text-blue-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              ></path>
            </svg>
          </div>
          <p className="mb-2 text-lg font-semibold text-gray-700">Drag & drop files here</p>
          <p className="mb-4 text-sm text-gray-500">
            or{" "}
            <span className="text-blue-600 hover:underline cursor-pointer" onClick={() => fileInputRef.current?.click()}>
              browse files
            </span>
          </p>
          <p className="text-xs text-gray-500">Supported formats: PDF, JPG, PNG, DICOM, etc.</p>
          <input
            ref={fileInputRef}
            type="file"
            className="hidden"
            onChange={handleFileChange}
            multiple
            accept=".pdf,.jpg,.jpeg,.png,.dcm,.dicom,.txt,.doc,.docx"
          />
        </div>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="mb-8">
          <h3 className="font-medium text-gray-700 mb-2">Uploaded Files:</h3>
          <ul className="space-y-2">
            {files.map((file, index) => (
              <li key={index} className="flex items-center justify-between p-3 bg-white border rounded-lg">
                <div className="flex items-center">
                  <div className="p-2 bg-blue-100 rounded">📄</div>
                  <div className="ml-3">
                    <p className="text-sm font-medium text-gray-700">{file.name}</p>
                    <p className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
                  </div>
                </div>
                <button onClick={() => removeFile(index)} className="text-red-500 hover:text-red-700">
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Symptom Section */}
      <div className="mb-8">
        <h3 className="font-medium text-gray-700 mb-3">Add Your Symptoms (Optional)</h3>
        <p className="text-sm text-gray-600 mb-4">
          Adding your symptoms will help our AI provide more accurate insights.
        </p>

        {/* Symptom Input */}
        <div className="flex mb-2">
          <input
            type="text"
            value={symptomInput}
            onChange={(e) => setSymptomInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSymptomAdd()}
            placeholder="Type a symptom..."
            className="flex-grow p-2 border border-gray-300 rounded-l focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button onClick={handleSymptomAdd} className="bg-blue-600 text-white py-2 px-4 rounded-r hover:bg-blue-700">
            Add
          </button>
        </div>

        {/* Common Symptoms */}
        <div className="mb-4">
          <p className="text-xs text-gray-600 mb-2">Common symptoms:</p>
          <div className="flex flex-wrap gap-2">
            {commonSymptoms.map((symptom, index) => (
              <button
                key={index}
                onClick={() => addCommonSymptom(symptom)}
                className="bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded-full hover:bg-gray-200"
              >
                + {symptom}
              </button>
            ))}
          </div>
        </div>

        {/* Selected Symptoms */}
        {symptoms.length > 0 && (
          <div>
            <p className="text-xs text-gray-600 mb-2">Your symptoms:</p>
            <div className="flex flex-wrap gap-2">
              {symptoms.map((symptom, index) => (
                <div key={index} className="flex items-center bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
                  <span className="text-sm">{symptom}</span>
                  <button onClick={() => removeSymptom(symptom)} className="ml-2 text-blue-600 hover:text-blue-800">
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Medical History Section */}
      <div className="mb-8">
        <h3 className="font-medium text-gray-700 mb-3">Medical History (Optional)</h3>
        <textarea
          placeholder="Enter any relevant medical history or additional information here..."
          value={medicalHistory}
          onChange={(e) => setMedicalHistory(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded h-24 focus:outline-none focus:ring-2 focus:ring-blue-500"
        ></textarea>
      </div>

      {/* Analyze Button */}
      <div className="flex justify-center">
        <button
          onClick={analyzeDiagnostics}
          className="bg-blue-600 text-white py-3 px-8 rounded-lg text-lg font-semibold hover:bg-blue-700 shadow-md"
        >
          Analyze Reports
        </button>
      </div>
    </div>
  );

  const renderResultsTab = () => (
    <div className="p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-6">Diagnostic Results</h2>

      {isAnalyzing ? (
        <div className="flex flex-col items-center justify-center py-10">
          <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
          <p className="text-gray-700 font-medium">Analyzing your medical reports...</p>
          <p className="text-sm text-gray-500 mt-2">This may take a few moments</p>
        </div>
      ) : results ? (
        <div>
          {/* Summary Card */}
          <div
            className={`p-4 rounded-lg mb-6 border-l-4 ${
              results.severity === "normal"
                ? "bg-green-50 border-green-500"
                : results.severity === "attention"
                ? "bg-yellow-50 border-yellow-500"
                : "bg-red-50 border-red-500"
            }`}
          >
            <div className="flex items-start">
              <div
                className={`p-2 rounded-full mr-3 ${
                  results.severity === "normal"
                    ? "bg-green-100 text-green-700"
                    : results.severity === "attention"
                    ? "bg-yellow-100 text-yellow-700"
                    : "bg-red-100 text-red-700"
                }`}
              >
                {results.severity === "normal" ? "✓" : results.severity === "attention" ? "!" : "!!"}
              </div>
              <div>
                <h3 className="font-bold text-gray-800">
                  {results.severity === "normal"
                    ? "Normal Results"
                    : results.severity === "attention"
                    ? "Requires Attention"
                    : "Critical Findings"}
                </h3>
                <p className="text-sm text-gray-600">
                  {results.severity === "normal"
                    ? "Your results are within normal ranges."
                    : results.severity === "attention"
                    ? "Your results require medical attention."
                    : "Your results indicate critical findings that need immediate medical attention."}
                </p>
                {results.followUp && (
                  <p className="text-sm font-medium mt-2">
                    Recommended follow-up: <span className="text-blue-700">{results.followUp}</span>
                  </p>
                )}
              </div>
            </div>
          </div>

          {/* Findings Section */}
          <div className="mb-6">
            <h3 className="text-xl font-semibold text-gray-800 mb-3">Key Findings</h3>
            <ul className="bg-white border rounded-lg p-4 space-y-2">
              {results.findings.map((finding, index) => (
                <li key={index} className="flex items-start">
                  <span className="text-blue-600 mr-2">•</span>
                  <span className="text-gray-700">{finding}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Recommendations Section */}
          <div className="mb-8">
            <h3 className="text-xl font-semibold text-gray-800 mb-3">Recommendations</h3>
            <ul className="bg-white border rounded-lg p-4 space-y-3">
              {results.recommendations.map((rec, index) => (
                <li key={index} className="flex items-start">
                  <div className="bg-blue-100 text-blue-800 w-6 h-6 rounded-full flex items-center justify-center mr-3 shrink-0">
                    {index + 1}
                  </div>
                  <span className="text-gray-700">{rec}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-wrap gap-4">
            <button
              onClick={saveToHistory}
              className="flex items-center bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700"
            >
              <span className="mr-2">💾</span> Save Results
            </button>
            <button className="flex items-center bg-white border border-blue-600 text-blue-600 py-2 px-4 rounded hover:bg-blue-50">
              <span className="mr-2">🖨️</span> Print Report
            </button>
            <button className="flex items-center bg-white border border-gray-300 text-gray-700 py-2 px-4 rounded hover:bg-gray-50">
              <span className="mr-2">✉️</span> Email Results
            </button>
          </div>
        </div>
      ) : (
        <div className="text-center py-10 text-gray-500">
          No results to display. Please upload medical reports and analyze them.
        </div>
      )}
    </div>
  );

  const renderHistoryTab = () => (
    <div className="p-6">
      <h2 className="text-2xl font-bold text-gray-800 mb-4">Saved Diagnostic History</h2>
      {history.length === 0 ? (
        <p className="text-gray-600">No saved results yet.</p>
      ) : (
        <ul className="space-y-4">
          {history.map((item, idx) => (
            <li key={idx} className="bg-white p-4 rounded border">
              <p>
                <strong>Severity:</strong> {item.severity}
              </p>
              <p className="mt-2 text-sm font-semibold">Findings:</p>
              <ul className="list-disc ml-6">
                {item.findings.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
              <p className="mt-2 text-sm font-semibold">Recommendations:</p>
              <ul className="list-disc ml-6">
                {item.recommendations.map((r, i) => (
                  <li key={i}>{r}</li>
                ))}
              </ul>
              {item.followUp && (
                <p className="mt-2">
                  <strong>Follow-up:</strong> {item.followUp}
                </p>
              )}
            </li>
          ))}
        </ul>
      )}
      <div className="text-center mt-6">
        <button
          onClick={() => setActiveTab("upload")}
          className="bg-blue-600 text-white py-2 px-6 rounded hover:bg-blue-700"
        >
          Start New Diagnostic
        </button>
      </div>
    </div>
  );

  return (
    <MainLayout>
      <div className="bg-white rounded-xl shadow-md overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-blue-800 p-4">
          <h1 className="text-2xl font-bold text-white">AI Diagnostic Assistant</h1>
          <p className="text-blue-100 text-sm">
            Upload your medical reports for AI-powered insights and recommendations
          </p>
        </div>

        {/* Tabs */}
        <div className="flex border-b">
          <button
            className={`flex-1 py-3 px-4 text-center ${
              activeTab === "upload"
                ? "border-b-2 border-blue-600 text-blue-600 font-medium"
                : "text-gray-600 hover:text-blue-500"
            }`}
            onClick={() => setActiveTab("upload")}
          >
            Upload Reports
          </button>
          <button
            className={`flex-1 py-3 px-4 text-center ${
              activeTab === "results"
                ? "border-b-2 border-blue-600 text-blue-600 font-medium"
                : "text-gray-600 hover:text-blue-500"
            }`}
            onClick={() => setActiveTab("results")}
          >
            Results
          </button>
          <button
            className={`flex-1 py-3 px-4 text-center ${
              activeTab === "history"
                ? "border-b-2 border-blue-600 text-blue-600 font-medium"
                : "text-gray-600 hover:text-blue-500"
            }`}
            onClick={() => setActiveTab("history")}
          >
            History
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === "upload" && renderUploadTab()}
        {activeTab === "results" && renderResultsTab()}
        {activeTab === "history" && renderHistoryTab()}
      </div>

      {/* Information Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
        <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-100">
          <div className="text-blue-600 text-lg mb-2">🔍 Advanced Analysis</div>
          <p className="text-sm text-gray-600">
            Our AI analyzes your reports using data from millions of medical cases, providing insights that might be missed
            in standard reviews.
          </p>
        </div>
        <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-100">
          <div className="text-blue-600 text-lg mb-2">🔒 Secure & Private</div>
          <p className="text-sm text-gray-600">
            Your medical data is encrypted and securely processed. We adhere to strict privacy standards and regulations.
          </p>
        </div>
        <div className="bg-white p-5 rounded-lg shadow-sm border border-gray-100">
          <div className="text-blue-600 text-lg mb-2">👩‍⚕️ Not a Replacement</div>
          <p className="text-sm text-gray-600">
            Our AI provides insights to help you and your doctor make informed decisions, but does not replace professional
            medical care.
          </p>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-8 p-4 border border-orange-200 bg-orange-50 rounded-lg">
        <div className="flex items-start">
          <div className="text-orange-500 mr-3">⚠️</div>
          <div>
            <h3 className="font-bold text-orange-800">Medical Disclaimer</h3>
            <p className="text-sm text-orange-700">
              The diagnostic information provided by MediMind AI is for informational purposes only and is not a substitute
              for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other
              qualified health provider with any questions you may have regarding a medical condition.
            </p>
          </div>
        </div>
      </div>
    </MainLayout>
  );
};

export default DiagnosticsPage;
