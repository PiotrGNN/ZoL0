
{ pkgs }: {
  deps = [
    pkgs.python310
    # Podstawowe narzędzia Python
    pkgs.python310Packages.pip
    pkgs.python310Packages.setuptools
    pkgs.python310Packages.wheel
    
    # Zależności do webowych aplikacji
    pkgs.python310Packages.flask
    pkgs.python310Packages.requests
    pkgs.python310Packages.python-dotenv
    
    # Analiza danych i ML
    pkgs.python310Packages.numpy
    pkgs.python310Packages.pandas
    pkgs.python310Packages.scikit-learn
    pkgs.python310Packages.matplotlib
    pkgs.python310Packages.xgboost
    pkgs.python310Packages.optuna
    
    # Trading API
    pkgs.python310Packages.ccxt
    pkgs.python310Packages.websocket-client
    
    # Narzędzia deweloperskie
    pkgs.python310Packages.pytest
    pkgs.python310Packages.black
    pkgs.python310Packages.flake8
    
    # Zależności systemowe
    pkgs.gcc
    pkgs.libffi
    pkgs.zlib
    pkgs.readline
    pkgs.openssl
    pkgs.sqlite
    pkgs.pkg-config
    
    # Dodatkowe pakiety dla matplotlib
    pkgs.freetype
    pkgs.cairo
    pkgs.gtk3
    pkgs.gobject-introspection
  ];
}
