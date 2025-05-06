Installation Guide
=================

This guide covers different ways to install and run the Trading System.

Docker Installation (Recommended)
-------------------------------

Prerequisites:

* Docker
* Docker Compose
* Git

Steps:

1. Clone the repository::

    git clone https://github.com/yourusername/trading-system.git
    cd trading-system

2. Build and start services::

    docker-compose up --build

3. Access the services:

   * Dashboard: http://localhost:8501
   * API: http://localhost:5000
   * Monitoring: http://localhost:3000

Manual Installation
-----------------

Prerequisites:

* Python 3.9+
* pip
* git
* virtualenv (recommended)

Basic Installation
~~~~~~~~~~~~~~~~

1. Clone the repository::

    git clone https://github.com/yourusername/trading-system.git
    cd trading-system

2. Create and activate virtual environment::

    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or
    venv\Scripts\activate     # Windows

3. Install dependencies::

    pip install -r requirements.txt

4. Initialize database::

    python -c "from data.db_manager import DatabaseManager; DatabaseManager('sqlite:///data/db/trading.db').init_db()"

5. Start services::

    python main.py      # API server
    python run.py       # Trading system
    streamlit run dashboard.py  # Dashboard

Development Installation
~~~~~~~~~~~~~~~~~~~~~~

For development, install additional dependencies::

    pip install -r requirements-dev.txt
    pre-commit install

This installs:

* Testing tools (pytest, coverage)
* Code quality tools (black, flake8, mypy)
* Documentation tools (Sphinx)
* Development utilities

Production Deployment
-------------------

For production deployment:

1. Configure environment::

    cp config/config.yaml.example config/config.yaml
    # Edit config.yaml with production settings

2. Set up SSL certificates::

    # Place SSL certificates in
    ./certs/fullchain.pem
    ./certs/privkey.pem

3. Deploy with Docker Compose::

    docker-compose -f docker-compose.prod.yml up -d

Security Considerations
--------------------

1. API Keys:
   * Store API keys in environment variables
   * Never commit API keys to version control
   * Use secure key management in production

2. Database:
   * Use strong passwords
   * Regular backups
   * Proper access controls

3. Network:
   * Configure firewalls
   * Use HTTPS/SSL
   * Implement rate limiting

Troubleshooting
-------------

Common Issues:

1. Database Connection::

    # Check database file permissions
    chmod 644 data/db/trading.db
    
    # Verify database exists
    python -c "from data.db_manager import DatabaseManager; DatabaseManager('sqlite:///data/db/trading.db').init_db()"

2. Port Conflicts::

    # Check if ports are in use
    lsof -i :8501
    lsof -i :5000

3. Dependencies::

    # Reinstall dependencies
    pip uninstall -r requirements.txt
    pip install -r requirements.txt