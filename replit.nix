
{ pkgs }: {
  deps = with pkgs; [
    python310
    python310Packages.pip
    python310Packages.setuptools
    python310Packages.wheel
    python310Packages.requests
    python310Packages.websockets
    python310Packages.uvloop
    python310Packages.orjson
    python310Packages.python-dotenv
    python310Packages.flask
    python310Packages.pandas
    python310Packages.numpy
    python310Packages.matplotlib
    python310Packages.pytz
    python310Packages.pillow
    python310Packages.scikit-learn
    python310Packages.python-dateutil
    python310Packages.nltk
    python310Packages.textblob
    python310Packages.pytest
    python310Packages.flake8
    python310Packages.black
    python310Packages.psutil
    curl
    git
    jq
    vim
  ];
  env = {
    PYTHONHOME = "${pkgs.python310}";
    PYTHONBIN = "${pkgs.python310}/bin/python3.10";
    LANG = "en_US.UTF-8";
    PYTHONIOENCODING = "utf-8";
    PYTHONPATH = "${pkgs.python310}/lib/python3.10/site-packages";
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ];
  };
}
