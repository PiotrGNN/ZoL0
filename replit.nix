
{ pkgs }: {
  deps = [
    # Podstawowe pakiety Python
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.setuptools
    pkgs.python310Packages.wheel

    # Środowisko Flask
    pkgs.python310Packages.flask
    pkgs.python310Packages.python-dotenv
    pkgs.python310Packages.requests
    pkgs.python310Packages.pybit

    # Analiza danych
    pkgs.python310Packages.numpy
    pkgs.python310Packages.pandas
    pkgs.python310Packages.matplotlib
    pkgs.python310Packages.scikit-learn
    
    # Narzędzia CLI
    pkgs.jq
    pkgs.curl
    pkgs.vim

    # Dodatkowe narzędzia, które wymieniłeś
    pkgs.python310Packages.websockets
    pkgs.python310Packages.uvloop
    pkgs.python310Packages.orjson
  ];
  
  # Konfiguracja środowiska 
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
