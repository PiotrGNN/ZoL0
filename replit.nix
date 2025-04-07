{ pkgs }: {
  deps = with pkgs; [
    python310
    python310Packages.requests
    python310Packages.websockets
    python310Packages.uvloop
    python310Packages.orjson
    python310Packages.python-dotenv
    python310Packages.setuptools
    python310Packages.flask
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
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ];
  };
}