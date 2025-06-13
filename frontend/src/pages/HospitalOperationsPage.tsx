import React, { useState, useEffect, useMemo } from 'react';
import { FC } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, LineChart, Line, ResponsiveContainer, Area, XAxis as RechartsXAxis, YAxis as RechartsYAxis } from 'recharts';
import { Users, Bed, AlertTriangle, UserCheck, FileText, Activity, TrendingUp, Calendar, Heart, Thermometer } from 'lucide-react';
import * as Papa from 'papaparse';
import { useNavigate } from 'react-router-dom';
import MainLayout from '../components/layout/MainLayout';


type Patient = {
  name: string;
  age: number;
  gender: string;
  bedNumber: string;
  bedType: 'general' | 'semi-private' | 'private';
  status: 'inpatient' | 'outpatient';
  priority: 'critical' | 'urgent' | 'normal';
  admissionDate: string;
  symptoms: string;
  medicalHistory: string;
  vitals: {
    temperature: number;
    heartRate: number;
    bloodPressure: string;
  };
  diagnosis: string;
  department: string;
};

type CsvPatientRow = {
  name: string;
  age: string; // CSV values come in as strings
  gender: string;
  bedNumber?: string;
  bedType?: string;
  status?: string;
  priority?: string;
  admissionDate?: string;
  symptoms?: string;
  medicalHistory?: string;
  vitals?: string; // Comes as stringified JSON
  diagnosis?: string;
  department?: string;
};

type PatientCardProps = {
  patient: Patient;
  onClick: (patient: Patient) => void;
};

type PatientModalProps = {
  patient: Patient;
  onClose: () => void;
};




const HospitalManagementAgent = () => {
  const navigate = useNavigate();
  const [csvData, setCsvData] = useState<Patient[]>([]);
  const [customPatient, setCustomPatient] = useState({
    name: '',
    age: '',
    gender: '',
    bedNumber: '',
    bedType: 'general',
    status: 'inpatient',
    priority: 'normal',
    admissionDate: '',
    symptoms: '',
    medicalHistory: ''
  });
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [showAddForm, setShowAddForm] = useState(false);
  const [activeView, setActiveView] = useState('dashboard');
  const [hospitalStats, setHospitalStats] = useState({
    totalBeds: 200,
    occupiedBeds: 150,
    availableBeds: 50,
    totalPatients: 150,
    unattendedPatients: 12,
    criticalPatients: 8,
    outpatients: 45
  });

  // Sample data for demonstration
  const sampleData: Patient[] = [
    {
      name: 'John Smith',
      age: 45,
      gender: 'Male',
      bedNumber: 'A101',
      bedType: 'general',
      status: 'inpatient',
      priority: 'critical',
      admissionDate: '2024-12-01',
      symptoms: 'Chest pain, shortness of breath, dizziness',
      medicalHistory: 'Hypertension, diabetes',
      vitals: { temperature: 38.2, heartRate: 95, bloodPressure: '140/90' },
      diagnosis: 'Acute coronary syndrome - requires immediate intervention',
      department: 'Cardiology'
    },
    {
      name: 'Sarah Johnson',
      age: 32,
      gender: 'Female',
      bedNumber: 'B205',
      bedType: 'general',
      status: 'inpatient',
      priority: 'normal',
      admissionDate: '2024-12-02',
      symptoms: 'Fever, headache, body aches',
      medicalHistory: 'No significant history',
      vitals: { temperature: 39.1, heartRate: 88, bloodPressure: '120/80' },
      diagnosis: 'Viral infection - monitor and symptomatic treatment',
      department: 'Internal Medicine'
    },
    {
      name: 'Michael Brown',
      age: 67,
      gender: 'Male',
      bedNumber: 'C302',
      bedType: 'general',
      status: 'outpatient',
      priority: 'urgent',
      admissionDate: '2024-12-03',
      symptoms: 'Severe abdominal pain, nausea',
      medicalHistory: 'Gallstones, previous surgery',
      vitals: { temperature: 37.8, heartRate: 102, bloodPressure: '135/85' },
      diagnosis: 'Possible gallbladder inflammation - requires imaging',
      department: 'Surgery'
    },
    {
      name: 'Emily Davis',
      age: 28,
      gender: 'Female',
      bedNumber: 'D101',
      bedType: 'general',
      status: 'inpatient',
      priority: 'normal',
      admissionDate: '2024-12-01',
      symptoms: 'Persistent cough, fatigue',
      medicalHistory: 'Asthma',
      vitals: { temperature: 37.2, heartRate: 78, bloodPressure: '110/70' },
      diagnosis: 'Respiratory infection - antibiotics prescribed',
      department: 'Pulmonology'
    }
  ];

  useEffect(() => {
    setCsvData(sampleData);
  }, []);

  const validBedTypes = ['general', 'semi-private', 'private'] as const;
  const validStatuses = ['inpatient', 'outpatient'] as const;
  const validPriorities = ['normal', 'urgent', 'critical'] as const;

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file && file.type === 'text/csv') {
      Papa.parse<CsvPatientRow>(file, {
        header: true,
        skipEmptyLines: true,
        complete: (result: Papa.ParseResult<CsvPatientRow>) => {
          const processedData: Patient[] = result.data
            .filter((row: CsvPatientRow) => row.name && row.name.trim() !== '')
            .map((row: CsvPatientRow): Patient => {
              return {
                name: row.name,
                age: parseInt(row.age) || 0,
                gender: row.gender,
                bedNumber: row.bedNumber || 'N/A',
                bedType: validBedTypes.includes(row.bedType as any)
                  ? (row.bedType as Patient['bedType'])
                  : 'general',
                status: validStatuses.includes(row.status as any)
                  ? (row.status as Patient['status'])
                  : 'inpatient',
                priority: validPriorities.includes(row.priority as any)
                  ? (row.priority as Patient['priority'])
                  : 'normal',
                admissionDate: row.admissionDate || new Date().toISOString().slice(0, 10),
                symptoms: row.symptoms || '',
                medicalHistory: row.medicalHistory || '',
                vitals: row.vitals
                  ? JSON.parse(row.vitals)
                  : { temperature: 37, heartRate: 72, bloodPressure: '120/80' },
                diagnosis: row.diagnosis || 'Pending diagnosis',
                department: row.department || 'General'
              };
            });

          setCsvData(processedData);
        }
      });
    }
  };


  const addCustomPatient = () => {
    if (customPatient.name.trim()) {
      const newPatient: Patient = {
        name: customPatient.name,
        age: parseInt(customPatient.age) || 0,
        gender: customPatient.gender || 'Not specified',
        bedNumber: customPatient.bedNumber || 'N/A',
        bedType: ['general', 'semi-private', 'private'].includes(customPatient.bedType)
          ? (customPatient.bedType as Patient['bedType'])
          : 'general',
        status: ['inpatient', 'outpatient'].includes(customPatient.status)
          ? (customPatient.status as Patient['status'])
          : 'inpatient',
        priority: ['normal', 'urgent', 'critical'].includes(customPatient.priority)
          ? (customPatient.priority as Patient['priority'])
          : 'normal',
        admissionDate: customPatient.admissionDate || new Date().toISOString().slice(0, 10),
        symptoms: customPatient.symptoms || '',
        medicalHistory: customPatient.medicalHistory || '',
        vitals: {
          temperature: 37,
          heartRate: 72,
          bloodPressure: '120/80'
        },
        diagnosis: 'Pending diagnosis - requires evaluation',
        department: 'General'
      };

      setCsvData([...csvData, newPatient]);

      setCustomPatient({
        name: '',
        age: '',
        gender: '',
        bedNumber: '',
        bedType: 'general',
        status: 'inpatient',
        priority: 'normal',
        admissionDate: '',
        symptoms: '',
        medicalHistory: ''
      });

      setShowAddForm(false);
    }
  };


  const getDashboardStats = useMemo(() => {
    const stats = csvData.reduce((acc, patient) => {
      acc.total++;
      if (patient.status === 'inpatient') acc.inpatients++;
      if (patient.status === 'outpatient') acc.outpatients++;
      if (patient.priority === 'critical') acc.critical++;
      if (patient.priority === 'urgent') acc.urgent++;
      return acc;
    }, { total: 0, inpatients: 0, outpatients: 0, critical: 0, urgent: 0 });

    return [
      { name: 'Total Patients', value: stats.total, color: '#3B82F6', icon: Users },
      { name: 'Inpatients', value: stats.inpatients, color: '#10B981', icon: Bed },
      { name: 'Outpatients', value: stats.outpatients, color: '#F59E0B', icon: UserCheck },
      { name: 'Critical Cases', value: stats.critical, color: '#EF4444', icon: AlertTriangle }
    ];
  }, [csvData]);

  const getPriorityDistribution = useMemo(() => {
    const priorityCount: Record<'normal' | 'urgent' | 'critical', number> = {
      normal: 0,
      urgent: 0,
      critical: 0
    };

    csvData.forEach((patient) => {
      priorityCount[patient.priority] += 1;
    });

    return Object.entries(priorityCount).map(([priority, count]) => ({
      name: priority.charAt(0).toUpperCase() + priority.slice(1),
      value: count,
      color: priority === 'critical' ? '#EF4444' : priority === 'urgent' ? '#F59E0B' : '#10B981'
    }));
  }, [csvData]);

  const getDepartmentStats = useMemo(() => {
    const deptCount: Record<string, number> = csvData.reduce((acc, patient) => {
      const dept = patient.department || 'General';
      acc[dept] = (acc[dept] || 0) + 1;
      return acc;
    }, {} as Record<string, number>); // 👈 assert the empty object type here

    return Object.entries(deptCount).map(([dept, count]) => ({
      name: dept,
      value: count
    }));
  }, [csvData]);

  const getAgeDistribution = useMemo(() => {
    const ageGroups = { '0-18': 0, '19-35': 0, '36-50': 0, '51-65': 0, '65+': 0 };

    csvData.forEach(patient => {
      const age = patient.age;
      if (age <= 18) ageGroups['0-18']++;
      else if (age <= 35) ageGroups['19-35']++;
      else if (age <= 50) ageGroups['36-50']++;
      else if (age <= 65) ageGroups['51-65']++;
      else ageGroups['65+']++;
    });

    return Object.entries(ageGroups).map(([range, count]) => ({
      ageRange: range,
      count
    }));
  }, [csvData]);

  const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'];

  const PatientCard: FC<PatientCardProps> = ({ patient, onClick }) => (
    <div
      className={`bg-white rounded-lg shadow-md p-4 cursor-pointer hover:shadow-lg transition-shadow border-l-4 ${patient.priority === 'critical' ? 'border-red-500' :
        patient.priority === 'urgent' ? 'border-yellow-500' : 'border-green-500'
        }`}
      onClick={() => onClick(patient)}
    >
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-semibold text-lg text-gray-800">{patient.name}</h3>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${patient.priority === 'critical' ? 'bg-red-100 text-red-800' :
          patient.priority === 'urgent' ? 'bg-yellow-100 text-yellow-800' :
            'bg-green-100 text-green-800'
          }`}>
          {patient.priority}
        </span>
      </div>
      <div className="space-y-1 text-sm text-gray-600">
        <p><strong>Age:</strong> {patient.age} | <strong>Gender:</strong> {patient.gender}</p>
        <p><strong>Bed:</strong> {patient.bedNumber || 'N/A'} | <strong>Status:</strong> {patient.status}</p>
        <p><strong>Department:</strong> {patient.department || 'General'}</p>
        <p className="text-xs mt-2 line-clamp-2">{patient.symptoms}</p>
      </div>
    </div>
  );

  const PatientModal: FC<PatientModalProps> = ({ patient, onClose }) => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-bold text-gray-800">{patient.name} - Medical Profile</h2>
            <button
              onClick={onClose}
              className="text-gray-500 hover:text-gray-700 text-xl font-bold"
            >
              ×
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="font-semibold text-blue-800 mb-2">Patient Information</h3>
                <div className="space-y-2 text-sm">
                  <p><strong>Age:</strong> {patient.age}</p>
                  <p><strong>Gender:</strong> {patient.gender}</p>
                  <p><strong>Bed Number:</strong> {patient.bedNumber || 'N/A'}</p>
                  <p><strong>Status:</strong> {patient.status}</p>
                  <p><strong>Priority:</strong> <span className={`font-medium ${patient.priority === 'critical' ? 'text-red-600' :
                    patient.priority === 'urgent' ? 'text-yellow-600' :
                      'text-green-600'
                    }`}>{patient.priority}</span></p>
                  <p><strong>Department:</strong> {patient.department}</p>
                  <p><strong>Admission Date:</strong> {patient.admissionDate}</p>
                </div>
              </div>

              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="font-semibold text-green-800 mb-2 flex items-center">
                  <Heart className="w-4 h-4 mr-2" />
                  Vital Signs
                </h3>
                <div className="space-y-2 text-sm">
                  <p><strong>Temperature:</strong> {patient.vitals?.temperature}°C</p>
                  <p><strong>Heart Rate:</strong> {patient.vitals?.heartRate} bpm</p>
                  <p><strong>Blood Pressure:</strong> {patient.vitals?.bloodPressure}</p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="bg-yellow-50 p-4 rounded-lg">
                <h3 className="font-semibold text-yellow-800 mb-2">Symptoms</h3>
                <p className="text-sm">{patient.symptoms}</p>
              </div>

              <div className="bg-purple-50 p-4 rounded-lg">
                <h3 className="font-semibold text-purple-800 mb-2">Medical History</h3>
                <p className="text-sm">{patient.medicalHistory}</p>
              </div>

              <div className="bg-red-50 p-4 rounded-lg">
                <h3 className="font-semibold text-red-800 mb-2 flex items-center">
                  <Activity className="w-4 h-4 mr-2" />
                  AI Diagnosis
                </h3>
                <p className="text-sm">{patient.diagnosis}</p>
                <button className="mt-2 bg-red-600 text-white px-4 py-2 rounded text-sm hover:bg-red-700 transition-colors">
                  Run Full Diagnostic Analysis
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );


  return (
    <MainLayout>
      <div className="min-h-screen bg-gray-50 p-6">
        <div className="max-w-7xl mx-auto">
          {/* Header */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <div className="flex justify-between items-center">
              <div>
                <h1 className="text-3xl font-bold text-gray-800 mb-2">Hospital Management System</h1>
                <p className="text-gray-600">Advanced patient management with AI-powered diagnostics</p>
              </div>
              <div className="flex space-x-4">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="csv-upload"
                />
                <label
                  htmlFor="csv-upload"
                  className="bg-blue-600 text-white px-4 py-2 rounded-lg cursor-pointer hover:bg-blue-700 transition-colors"
                >
                  Upload CSV
                </label>
                <button
                  onClick={() => setShowAddForm(true)}
                  className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
                >
                  Add Patient
                </button>
              </div>
            </div>
          </div>

          {/* Navigation */}
          <div className="bg-white rounded-lg shadow-md p-4 mb-6">
            <div className="flex space-x-4">
              {['dashboard', 'patients', 'analytics'].map((view) => (
                <button
                  key={view}
                  onClick={() => setActiveView(view)}
                  className={`px-4 py-2 rounded-lg capitalize font-medium transition-colors ${activeView === view
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-600 hover:bg-gray-100'
                    }`}
                >
                  {view}
                </button>
              ))}
            </div>
          </div>

          {/* Dashboard View */}
          {activeView === 'dashboard' && (
            <>
              {/* Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
                {getDashboardStats.map((stat, index) => {
                  const IconComponent = stat.icon;
                  return (
                    <div key={index} className="bg-white rounded-lg shadow-md p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-gray-600 text-sm">{stat.name}</p>
                          <p className="text-3xl font-bold" style={{ color: stat.color }}>
                            {stat.value}
                          </p>
                        </div>
                        <IconComponent className="w-8 h-8" style={{ color: stat.color }} />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                {/* Priority Distribution */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold mb-4">Patient Priority Distribution</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={getPriorityDistribution}
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        dataKey="value"
                        label={({ name, value }) => `${name}: ${value}`}
                      >
                        {getPriorityDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Department Statistics */}
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold mb-4">Patients by Department</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={getDepartmentStats}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="department" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="patients" fill="#3B82F6" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Age Distribution */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-4">Age Distribution</h3>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={getAgeDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="ageRange" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#10B981" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}

          {/* Patients View */}
          {activeView === 'patients' && (
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-lg font-semibold">Patient List ({csvData.length})</h3>
                <div className="flex space-x-2">
                  <select className="border rounded px-3 py-1 text-sm">
                    <option>All Priorities</option>
                    <option>Critical</option>
                    <option>Urgent</option>
                    <option>Normal</option>
                  </select>
                  <select className="border rounded px-3 py-1 text-sm">
                    <option>All Departments</option>
                    <option>Cardiology</option>
                    <option>Surgery</option>
                    <option>Internal Medicine</option>
                  </select>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {csvData.map((patient, index) => (
                  <PatientCard
                    key={index}
                    patient={patient}
                    onClick={(p: Patient) => setSelectedPatient(p)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Analytics View */}
          {activeView === 'analytics' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold mb-4">Monthly Admissions Trend</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <LineChart data={[
                      { month: 'Jan', admissions: 120 },
                      { month: 'Feb', admissions: 135 },
                      { month: 'Mar', admissions: 148 },
                      { month: 'Apr', admissions: 162 },
                      { month: 'May', admissions: 178 },
                      { month: 'Jun', admissions: 155 }
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="month" />
                      <YAxis />
                      <Tooltip />
                      <Line type="monotone" dataKey="admissions" stroke="#3B82F6" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="bg-white rounded-lg shadow-md p-6">
                  <h3 className="text-lg font-semibold mb-4">Bed Occupancy Rate</h3>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span>ICU</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div className="bg-red-500 h-2 rounded-full" style={{ width: '85%' }}></div>
                        </div>
                        <span className="text-sm">85%</span>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>General</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div className="bg-yellow-500 h-2 rounded-full" style={{ width: '72%' }}></div>
                        </div>
                        <span className="text-sm">72%</span>
                      </div>
                    </div>
                    <div className="flex justify-between items-center">
                      <span>Emergency</span>
                      <div className="flex items-center space-x-2">
                        <div className="w-32 bg-gray-200 rounded-full h-2">
                          <div className="bg-green-500 h-2 rounded-full" style={{ width: '45%' }}></div>
                        </div>
                        <span className="text-sm">45%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Add Patient Modal */}
          {showAddForm && (
            <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
              <div className="bg-white rounded-lg max-w-md w-full mx-4">
                <div className="p-6">
                  <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold">Add New Patient</h3>
                    <button
                      onClick={() => setShowAddForm(false)}
                      className="text-gray-500 hover:text-gray-700"
                    >
                      ×
                    </button>
                  </div>
                  <div className="space-y-4">
                    <input
                      type="text"
                      placeholder="Patient Name"
                      value={customPatient.name}
                      onChange={(e) => setCustomPatient({ ...customPatient, name: e.target.value })}
                      className="w-full border rounded px-3 py-2"
                    />
                    <div className="grid grid-cols-2 gap-4">
                      <input
                        type="number"
                        placeholder="Age"
                        value={customPatient.age}
                        onChange={(e) => setCustomPatient({ ...customPatient, age: e.target.value })}
                        className="w-full border rounded px-3 py-2"
                      />
                      <select
                        value={customPatient.gender}
                        onChange={(e) => setCustomPatient({ ...customPatient, gender: e.target.value })}
                        className="w-full border rounded px-3 py-2"
                      >
                        <option value="">Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <select
                        value={customPatient.status}
                        onChange={(e) => setCustomPatient({ ...customPatient, status: e.target.value })}
                        className="w-full border rounded px-3 py-2"
                      >
                        <option value="inpatient">Inpatient</option>
                        <option value="outpatient">Outpatient</option>
                      </select>
                      <select
                        value={customPatient.priority}
                        onChange={(e) => setCustomPatient({ ...customPatient, priority: e.target.value })}
                        className="w-full border rounded px-3 py-2"
                      >
                        <option value="normal">Normal</option>
                        <option value="urgent">Urgent</option>
                        <option value="critical">Critical</option>
                      </select>
                    </div>
                    <input
                      type="date"
                      value={customPatient.admissionDate}
                      onChange={(e) => setCustomPatient({ ...customPatient, admissionDate: e.target.value })}
                      className="w-full border rounded px-3 py-2"
                    />
                    <textarea
                      placeholder="Symptoms"
                      value={customPatient.symptoms}
                      onChange={(e) => setCustomPatient({ ...customPatient, symptoms: e.target.value })}
                      className="w-full border rounded px-3 py-2 h-20"
                    />
                    <textarea
                      placeholder="Medical History"
                      value={customPatient.medicalHistory}
                      onChange={(e) => setCustomPatient({ ...customPatient, medicalHistory: e.target.value })}
                      className="w-full border rounded px-3 py-2 h-20"
                    />
                  </div>
                  <div className="flex justify-end space-x-4 mt-6">
                    <button
                      onClick={() => setShowAddForm(false)}
                      className="px-4 py-2 text-gray-600 hover:text-gray-800"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={addCustomPatient}
                      className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                    >
                      Add Patient
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Patient Detail Modal */}
          {selectedPatient && (
            <PatientModal
              patient={selectedPatient}
              onClose={() => setSelectedPatient(null)}
            />
          )}
        </div>
      </div>
    </MainLayout>
  );
};

export default HospitalManagementAgent;