import type { FC } from 'react';
import { useNavigate } from 'react-router-dom';
import MainLayout from '../components/layout/MainLayout';

const HospitalOperationsPage: FC = () => {
  const navigate = useNavigate();

  return (
    <MainLayout>
      {/* Two Column Layout */}
      <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Patient Records Column */}
        <section className="bg-white p-6 rounded shadow">
          <div className="flex items-center text-blue-600 font-bold text-xl mb-4">
            <span className="mr-2">👤</span> Patient Records
          </div>
          <p className="text-sm text-gray-600 mb-6">
            Manage patient records and visualize healthcare data for better insights.
          </p>

          {/* File Upload Box */}
          <div className="mt-4">
            <p className="text-xs text-gray-500 mb-2">Upload patient records (CSV)</p>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 bg-gray-50">
              <div className="flex flex-col items-center">
                <div className="rounded-full bg-gray-200 p-3 mb-2">
                  <span className="text-gray-600">⬆️</span>
                </div>
                <p className="text-sm text-gray-600">Drag and drop file here</p>
                <p className="text-xs text-gray-500 mt-1">Limit 200MB per file • CSV</p>
              </div>
              <button className="mt-4 w-full bg-blue-500 text-white py-2 px-4 rounded hover:bg-blue-600">
                Browse files
              </button>
            </div>
          </div>
        </section>

        {/* Data Visualizations Column */}
        <section className="bg-white p-6 rounded shadow">
          <div className="flex items-center text-blue-600 font-bold text-xl mb-4">
            <span className="mr-2">📊</span> Data Visualizations
          </div>
          <p className="text-sm text-gray-600 mb-6">
            Visual insights from patient records to aid in healthcare decision making.
          </p>

          {/* Visualization placeholder */}
          <div className="mt-4">
            <p className="text-xs text-gray-500 mb-2">Upload patient data and generate visualizations to see insights here</p>
            <div className="border border-gray-300 rounded-lg p-6 bg-blue-50">
              <p className="text-center text-gray-500">No visualizations yet</p>
            </div>
          </div>
        </section>
      </div>

      {/* Contact Us */}
      <section className="mt-10 bg-white p-6 rounded shadow">
        <div className="text-blue-600 font-bold text-lg">📞 Contact Us -</div>
        <p className="text-md text-gray-700 mt-1">
          Have questions? Reach out to our support team for assistance.
        </p>
        <p className="text-xl font-semibold text-gray-900 mt-2">
          9823123139
        </p>
      </section>
    </MainLayout>
  );
};

export default HospitalOperationsPage;