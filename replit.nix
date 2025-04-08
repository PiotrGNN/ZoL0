
{ pkgs }: {
  deps = with pkgs; [
    # Podstawowe pakiety Python
    python310
    python310Packages.pip
    python310Packages.setuptools
    python310Packages.wheel

    # Zmienne środowiskowe i Flask
    python310Packages.python-dotenv
    python310Packages.flask
    python310Packages.werkzeug
    python310Packages.jinja2

    # Klient giełdy i obsługa API
    python310Packages.requests
    python310Packages.websocket-client
    python310Packages.pybit

    # Analiza danych
    python310Packages.numpy
    python310Packages.pandas
    python310Packages.matplotlib
    python310Packages.scikit-learn
    python310Packages.pillow
    python310Packages.pytz
    python310Packages.python-dateutil
    
    # Optymalizacja i przetwarzanie
    python310Packages.xgboost
    python310Packages.optuna
    python310Packages.orjson
    python310Packages.uvloop
    python310Packages.websockets

    # Narzędzia deweloperskie
    python310Packages.pytest
    python310Packages.flake8
    python310Packages.black
    
    # Narzędzia systemowe
    curl
    git
    jq
    vim
    psutil
  ];
  
  env = {
    PYTHONHOME = "${pkgs.python310}";
    PYTHONBIN = "${pkgs.python310}/bin/python3.10";
    LANG = "en_US.UTF-8";
    PYTHONIOENCODING = "utf-8";
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ];
  };
}
