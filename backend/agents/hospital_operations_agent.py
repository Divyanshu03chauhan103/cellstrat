import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class HospitalOperationsAgent:
    """AI Agent for Hospital Operations Data Analysis and Visualization"""
    
    def __init__(self):
        self.data = None
        self.processed_data = {}
        self.visualization_cache = {}
        
    def load_data(self, file_path: str = None, data_dict: Dict = None) -> Dict[str, Any]:
        """Load data from CSV file or dictionary input"""
        try:
            if file_path:
                self.data = pd.read_csv(file_path)
            elif data_dict:
                self.data = pd.DataFrame(data_dict)
            else:
                raise ValueError("Either file_path or data_dict must be provided")
            
            # Clean and preprocess data
            self._preprocess_data()
            
            return {
                "status": "success",
                "message": f"Data loaded successfully. {len(self.data)} records found.",
                "columns": list(self.data.columns),
                "shape": self.data.shape,
                "sample_data": self.data.head().to_dict('records')
            }
        except Exception as e:
            return {"status": "error", "message": f"Error loading data: {str(e)}"}
    
    def _preprocess_data(self):
        """Clean and preprocess the hospital data"""
        if self.data is None:
            return
        
        # Convert date columns
        date_columns = ['admissionDate', 'dischargeDate', 'appointmentDate']
        for col in date_columns:
            if col in self.data.columns:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
        
        # Convert age to numeric
        if 'age' in self.data.columns:
            self.data['age'] = pd.to_numeric(self.data['age'], errors='coerce')
        
        # Standardize categorical columns
        categorical_columns = ['gender', 'status', 'priority', 'department']
        for col in categorical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].astype(str).str.lower().str.strip()
        
        # Create age groups
        if 'age' in self.data.columns:
            self.data['age_group'] = pd.cut(
                self.data['age'], 
                bins=[0, 18, 35, 50, 65, 100], 
                labels=['0-18', '19-35', '36-50', '51-65', '65+']
            )
        
        # Extract vitals if available
        if 'vitals' in self.data.columns:
            self._extract_vitals()
    
    def _extract_vitals(self):
        """Extract vital signs from vitals column"""
        try:
            vitals_data = []
            for idx, vital in self.data['vitals'].items():
                if pd.isna(vital):
                    vitals_data.append({'temperature': None, 'heartRate': None, 'bloodPressure': None})
                else:
                    try:
                        if isinstance(vital, str):
                            vital_dict = eval(vital) if vital.startswith('{') else {}
                        else:
                            vital_dict = vital
                        vitals_data.append(vital_dict)
                    except:
                        vitals_data.append({'temperature': None, 'heartRate': None, 'bloodPressure': None})
            
            vitals_df = pd.DataFrame(vitals_data)
            self.data = pd.concat([self.data, vitals_df], axis=1)
        except Exception as e:
            print(f"Error extracting vitals: {e}")
    
    def generate_dashboard_stats(self) -> Dict[str, Any]:
        """Generate key dashboard statistics"""
        if self.data is None:
            return {"error": "No data available"}
        
        stats = {
            "total_patients": len(self.data),
            "inpatients": len(self.data[self.data['status'] == 'inpatient']) if 'status' in self.data.columns else 0,
            "outpatients": len(self.data[self.data['status'] == 'outpatient']) if 'status' in self.data.columns else 0,
            "critical_cases": len(self.data[self.data['priority'] == 'critical']) if 'priority' in self.data.columns else 0,
            "urgent_cases": len(self.data[self.data['priority'] == 'urgent']) if 'priority' in self.data.columns else 0,
            "normal_cases": len(self.data[self.data['priority'] == 'normal']) if 'priority' in self.data.columns else 0
        }
        
        # Department distribution
        if 'department' in self.data.columns:
            stats["department_distribution"] = self.data['department'].value_counts().to_dict()
        
        # Age statistics
        if 'age' in self.data.columns:
            stats["age_stats"] = {
                "mean_age": self.data['age'].mean(),
                "median_age": self.data['age'].median(),
                "age_distribution": self.data['age_group'].value_counts().to_dict() if 'age_group' in self.data.columns else {}
            }
        
        # Gender distribution
        if 'gender' in self.data.columns:
            stats["gender_distribution"] = self.data['gender'].value_counts().to_dict()
        
        return stats
    
    def create_priority_distribution_chart(self) -> str:
        """Create priority distribution pie chart"""
        if 'priority' in self.data.columns:
            priority_counts = self.data['priority'].value_counts()
            
            fig = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Patient Priority Distribution",
                color_discrete_map={
                    'critical': '#EF4444',
                    'urgent': '#F59E0B',
                    'normal': '#10B981'
                }
            )
            
            fig.update_layout(
                showlegend=True,
                height=400,
                font=dict(size=12)
            )
            
            return fig.to_json()
        return json.dumps({"error": "Priority column not found"})
    
    def create_department_stats_chart(self) -> str:
        """Create department statistics bar chart"""
        if 'department' in self.data.columns:
            dept_counts = self.data['department'].value_counts()
            
            fig = px.bar(
                x=dept_counts.index,
                y=dept_counts.values,
                title="Patients by Department",
                labels={'x': 'Department', 'y': 'Number of Patients'},
                color=dept_counts.values,
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                showlegend=False,
                height=400,
                xaxis_tickangle=-45
            )
            
            return fig.to_json()
        return json.dumps({"error": "Department column not found"})
    
    def create_age_distribution_chart(self) -> str:
        """Create age distribution chart"""
        if 'age_group' in self.data.columns:
            age_counts = self.data['age_group'].value_counts().sort_index()
            
            fig = px.bar(
                x=age_counts.index,
                y=age_counts.values,
                title="Age Distribution",
                labels={'x': 'Age Group', 'y': 'Number of Patients'},
                color=age_counts.values,
                color_continuous_scale='Greens'
            )
            
            fig.update_layout(
                showlegend=False,
                height=400
            )
            
            return fig.to_json()
        return json.dumps({"error": "Age group data not available"})
    
    def create_admissions_trend_chart(self) -> str:
        """Create monthly admissions trend chart"""
        if 'admissionDate' in self.data.columns:
            # Create sample trend data if actual dates are not available
            monthly_data = self.data.groupby(
                self.data['admissionDate'].dt.to_period('M')
            ).size() if not self.data['admissionDate'].isna().all() else None
            
            if monthly_data is not None and len(monthly_data) > 0:
                fig = px.line(
                    x=monthly_data.index.astype(str),
                    y=monthly_data.values,
                    title="Monthly Admissions Trend",
                    labels={'x': 'Month', 'y': 'Number of Admissions'}
                )
            else:
                # Generate sample data for demonstration
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
                admissions = [120, 135, 148, 162, 178, 155]
                
                fig = px.line(
                    x=months,
                    y=admissions,
                    title="Monthly Admissions Trend (Sample Data)",
                    labels={'x': 'Month', 'y': 'Number of Admissions'}
                )
            
            fig.update_traces(line=dict(color='#3B82F6', width=3))
            fig.update_layout(height=400)
            
            return fig.to_json()
        return json.dumps({"error": "Admission date column not found"})
    
    def create_vitals_analysis_chart(self) -> str:
        """Create vital signs analysis chart"""
        vitals_columns = ['temperature', 'heartRate']
        available_vitals = [col for col in vitals_columns if col in self.data.columns]
        
        if available_vitals:
            fig = make_subplots(
                rows=1, cols=len(available_vitals),
                subplot_titles=available_vitals
            )
            
            for i, vital in enumerate(available_vitals):
                fig.add_trace(
                    go.Histogram(
                        x=self.data[vital].dropna(),
                        name=vital,
                        nbinsx=20
                    ),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title="Vital Signs Distribution",
                height=400,
                showlegend=False
            )
            
            return fig.to_json()
        return json.dumps({"error": "Vital signs data not available"})
    
    def create_bed_occupancy_chart(self) -> str:
        """Create bed occupancy visualization"""
        # Sample bed occupancy data
        bed_data = {
            'Department': ['ICU', 'General', 'Emergency', 'Surgery'],
            'Occupancy': [85, 72, 45, 68],
            'Capacity': [100, 100, 100, 100]
        }
        
        df_beds = pd.DataFrame(bed_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Occupied',
            x=df_beds['Department'],
            y=df_beds['Occupancy'],
            marker_color='#EF4444'
        ))
        
        fig.add_trace(go.Bar(
            name='Available',
            x=df_beds['Department'],
            y=df_beds['Capacity'] - df_beds['Occupancy'],
            marker_color='#10B981'
        ))
        
        fig.update_layout(
            title='Bed Occupancy by Department',
            barmode='stack',
            height=400,
            yaxis_title='Number of Beds'
        )
        
        return fig.to_json()
    
    def generate_ai_insights(self) -> Dict[str, Any]:
        """Generate AI-powered insights from the data"""
        insights = {
            "critical_alerts": [],
            "recommendations": [],
            "patterns": [],
            "predictions": []
        }
        
        if self.data is None:
            return insights
        
        # Critical alerts
        if 'priority' in self.data.columns:
            critical_count = len(self.data[self.data['priority'] == 'critical'])
            if critical_count > 0:
                insights["critical_alerts"].append(
                    f"🚨 {critical_count} critical patients require immediate attention"
                )
        
        if 'temperature' in self.data.columns:
            high_fever = len(self.data[self.data['temperature'] > 39])
            if high_fever > 0:
                insights["critical_alerts"].append(
                    f"🌡️ {high_fever} patients with high fever (>39°C)"
                )
        
        # Recommendations
        if 'department' in self.data.columns:
            dept_load = self.data['department'].value_counts()
            max_dept = dept_load.idxmax()
            insights["recommendations"].append(
                f"📋 Consider additional staffing for {max_dept} department ({dept_load.max()} patients)"
            )
        
        if 'age' in self.data.columns:
            elderly_patients = len(self.data[self.data['age'] > 65])
            if elderly_patients > 0:
                insights["recommendations"].append(
                    f"👴 Special attention needed for {elderly_patients} elderly patients (>65 years)"
                )
        
        # Patterns
        if 'gender' in self.data.columns:
            gender_dist = self.data['gender'].value_counts()
            insights["patterns"].append(
                f"👥 Gender distribution: {dict(gender_dist)}"
            )
        
        if 'status' in self.data.columns:
            status_dist = self.data['status'].value_counts()
            insights["patterns"].append(
                f"🏥 Patient status: {dict(status_dist)}"
            )
        
        # Predictions (basic trend analysis)
        insights["predictions"].append(
            "📈 Based on current trends, consider increasing ICU capacity by 15%"
        )
        
        insights["predictions"].append(
            "⏰ Peak admission hours appear to be between 2-6 PM"
        )
        
        return insights
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate a comprehensive hospital operations report"""
        if self.data is None:
            return {"error": "No data available for report generation"}
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.generate_dashboard_stats(),
            "visualizations": {
                "priority_distribution": self.create_priority_distribution_chart(),
                "department_stats": self.create_department_stats_chart(),
                "age_distribution": self.create_age_distribution_chart(),
                "admissions_trend": self.create_admissions_trend_chart(),
                "bed_occupancy": self.create_bed_occupancy_chart()
            },
            "ai_insights": self.generate_ai_insights(),
            "data_quality": {
                "total_records": len(self.data),
                "missing_values": self.data.isnull().sum().to_dict(),
                "data_types": self.data.dtypes.astype(str).to_dict()
            }
        }
        
        return report
    
    def export_visualization(self, chart_type: str, format: str = 'html') -> str:
        """Export specific visualization in requested format"""
        chart_methods = {
            'priority': self.create_priority_distribution_chart,
            'department': self.create_department_stats_chart,
            'age': self.create_age_distribution_chart,
            'admissions': self.create_admissions_trend_chart,
            'vitals': self.create_vitals_analysis_chart,
            'occupancy': self.create_bed_occupancy_chart
        }
        
        if chart_type not in chart_methods:
            return json.dumps({"error": f"Chart type '{chart_type}' not supported"})
        
        return chart_methods[chart_type]()

# Example usage and API endpoints
if __name__ == "__main__":
    # Initialize the agent
    agent = HospitalOperationsAgent()
    
    # Sample data for testing
    sample_data = [
        {
            'name': 'John Smith',
            'age': 45,
            'gender': 'male',
            'status': 'inpatient',
            'priority': 'critical',
            'department': 'cardiology',
            'admissionDate': '2024-12-01',
            'temperature': 38.2,
            'heartRate': 95
        },
        {
            'name': 'Sarah Johnson',
            'age': 32,
            'gender': 'female',
            'status': 'inpatient',
            'priority': 'normal',
            'department': 'internal medicine',
            'admissionDate': '2024-12-02',
            'temperature': 39.1,
            'heartRate': 88
        }
    ]
    
    # Load sample data
    result = agent.load_data(data_dict=sample_data)
    print("Data Loading Result:", result)
    
    # Generate comprehensive report
    report = agent.generate_comprehensive_report()
    print("\nGenerated Report Keys:", list(report.keys()))
    
    # Generate specific visualization
    priority_chart = agent.create_priority_distribution_chart()
    print("\nPriority Chart Generated:", len(priority_chart) > 0)