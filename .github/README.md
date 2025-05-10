# Financial Security System

A comprehensive financial security system that provides real-time transaction monitoring, fraud detection, and credit risk assessment capabilities.

## System Components

The system consists of several microservices working together to provide a complete financial security solution:

1. **Backend API Service**

    - RESTful API service built with FastAPI
    - Handles core business logic and service orchestration
    - Runs on port 8000

2. **Transaction Anomaly Detection**

    - Real-time transaction monitoring and anomaly detection
    - Uses machine learning to identify suspicious patterns
    - Integrates with Kafka for event streaming

3. **Fraud Network Detection**

    - Network analysis for fraud detection
    - Uses graph database (Neo4j) for relationship analysis
    - Identifies complex fraud patterns and networks

4. **Credit Risk Scoring**

    - Credit risk assessment and scoring
    - Real-time risk evaluation
    - Historical data analysis

5. **Data Visualization (Apache Superset)**
    - Interactive dashboards and visualizations
    - Real-time monitoring of system metrics
    - Accessible at port 8088

## Technology Stack

-   **Message Broker**: Apache Kafka
-   **Databases**:
    -   PostgreSQL (Main database)
    -   Neo4j (Graph database for fraud network analysis)
    -   ClickHouse (Analytics database)
-   **Data Visualization**: Apache Superset
-   **Container Orchestration**: Docker Compose
-   **API Framework**: FastAPI
-   **Development Tools**:
    -   Black (Code formatting)
    -   Pre-commit hooks
    -   Flake8 (Linting)

## Prerequisites

-   Docker and Docker Compose
-   Pyenv with Python 3.10.x, 3.12.x and 3.13.x
-   Git

## Getting Started

1. Clone the repository:

    ```bash
    git clone https://github.com/Financial-Fraud-Detection-System/financial-security-system.git
    cd financial-security-system
    ```

2. Start the services:

    ```bash
    docker-compose up -d
    ```

3. Access the services:
    - Backend API: http://localhost:8000
    - Superset Dashboard: http://localhost:8088
    - Neo4j Browser: http://localhost:7474

## Development Setup

1. Synchronise virtual environment across all directories (from the root directory):

    ```bash
    ./scripts/sync_venv.sh
    # On Windows: ./scripts/Sync_Venv.ps1
    ```

2. Install pre-commit hooks (From the root directory activate the virtual environment and run):

    ```bash
    pre-commit install
    ```

3. To run tests:
    ```bash
    ./scripts/run_all_tests.sh
    # On Windows: ./scripts/run_all_tests.ps1
    ```

## Project Structure

```
financial-security-system/
├── backend/                 # FastAPI backend service
├── transaction_anomaly_detection/  # Transaction monitoring service
├── fraud_network_detection/ # Fraud detection service
├── credit_risk_scoring/     # Credit risk assessment service
├── superset/               # Data visualization configuration
├── libs/                   # Shared libraries
├── scripts/                # Utility scripts
└── docker-compose.yaml     # Service orchestration
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Security

This system handles sensitive financial data. Please ensure proper security measures are in place before deployment to production.
