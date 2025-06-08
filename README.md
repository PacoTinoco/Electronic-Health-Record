# 🏥 Rural Electronic Health Record (EMR) Analytics System

## 📋 Overview

This project analyzes and visualizes healthcare data from rural health centers in Michoacán, Mexico. It provides actionable insights for improving healthcare delivery in underserved rural communities through an interactive dashboard that monitors patient care, resource management, and epidemiological trends.

## 🎯 Key Features

- **Real-time Analytics Dashboard**: Interactive visualization of health center operations
- **Patient Segmentation**: Risk-based classification considering distance and chronic conditions
- **Resource Management**: Medication inventory tracking and staff workload analysis
- **Epidemiological Insights**: Disease pattern analysis and consultation trends
- **Connectivity Monitoring**: Track internet connectivity issues affecting EMR usage
- **Alert System**: Automated warnings for critical medication shortages and high-risk patients

## 📊 Dashboard Sections

### 1. **Main KPIs**
- Total patients and active status
- Consultation volume and trends
- Medical staff availability
- Critical medication alerts
- High-risk patient count

### 2. **Epidemiological Analysis**
- Top 10 most frequent diagnoses
- Monthly consultation trends
- Disease pattern visualization

### 3. **Resource Management**
- Medication inventory status (🔴 Critical, 🟡 Alert, 🟢 OK)
- Staff workload gauge
- Connectivity monitoring charts

### 4. **Geographic & Segmentation Analysis**
- Patient distribution by distance
- Risk-accessibility matrix
- Rural coverage insights

### 5. **Alerts & Recommendations**
- Critical stock warnings
- High-risk patient alerts
- Actionable recommendations

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/PacoTinoco/Electronic-Health-Record.git
cd Electronic-Health-Record
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run dashboard_emr_rural.py
```

4. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
Electronic-Health-Record/
│
├── data/                          # Healthcare datasets
│   ├── pacientes_rural.csv        # Rural patient registry
│   ├── historias_clinicas.csv     # Clinical history records
│   ├── prescripciones.csv         # Prescription data
│   ├── inventario_medicamentos.csv # Medication inventory
│   ├── personal_medico.csv        # Medical staff information
│   ├── conectividad.csv           # Connectivity logs
│   └── pacientes_segmentados.csv  # Segmented patient data
│
├── scripts/                       # Analysis scripts
│   └── health_ehr.ipynb          # Jupyter notebook analysis
│
├── visualizations/               # Generated charts and graphs
│   ├── analisis_consultas.png
│   ├── analisis_demanda_recursos.png
│   └── segmentacion_pacientes.png
│
├── dashboard_emr_rural.py        # Main Streamlit dashboard
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
```

## 📈 Data Description

### Patient Data (`pacientes_rural.csv`)
- **10,000 patients** from 10 rural health centers
- Demographics, insurance status, distance to health center
- Chronic condition indicators

### Clinical Records (`historias_clinicas.csv`)
- **50,000 consultation records**
- Diagnoses using ICD-10 classification
- Temporal patterns and seasonal trends

### Medication Inventory (`inventario_medicamentos.csv`)
- Real-time stock levels
- Critical shortage detection
- Supply chain insights

### Connectivity Data (`conectividad.csv`)
- Daily connectivity hours per health center
- Impact on EMR system usage
- Infrastructure challenges

## 🔧 Technologies Used

- **Python 3.9+**: Core programming language
- **Streamlit**: Interactive web dashboard framework
- **Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **NumPy**: Numerical computations

## 📊 Key Insights

1. **High-Risk Patients**: Identification of patients with chronic conditions living >20km from health centers
2. **Resource Optimization**: Real-time tracking prevents medication stockouts
3. **Disease Patterns**: Seasonal trends in respiratory and gastrointestinal conditions
4. **Connectivity Issues**: Average 8-12 hours daily connectivity affects EMR adoption

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Paco Tinoco** - *Initial work* - [PacoTinoco](https://github.com/PacoTinoco)

## 🙏 Acknowledgments

- Rural health centers in Michoacán for their collaboration
- Mexican Ministry of Health for supporting rural healthcare initiatives
- Open-source community for the amazing tools

## 📞 Contact

For questions or suggestions, please open an issue or contact:
- Email: jose.tinoco@iteso.mx
- LinkedIn: [LinkedIn](https://www.linkedin.com/in/jos%C3%A9-francisco-tinoco-ceja-908681265/)

---

**Note**: This project uses synthetic data for demonstration purposes. All patient information is fictional and any resemblance to real persons is purely coincidental.