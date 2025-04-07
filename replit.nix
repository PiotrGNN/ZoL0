{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.flask
    pkgs.python310Packages.python-dotenv
    pkgs.python310Packages.requests
    pkgs.python310Packages.pandas
    pkgs.python310Packages.numpy
    pkgs.python310Packages.matplotlib
    pkgs.python310Packages.scikit-learn
    pkgs.python310Packages.pytest
  ];
  env = {
    PYTHONBIN = "${pkgs.python310}/bin/python3.10";
    LANG = "en_US.UTF-8";
  };
}