
{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.flask
    pkgs.python310Packages.python-dotenv
    pkgs.python310Packages.pandas
    pkgs.python310Packages.numpy
    pkgs.python310Packages.matplotlib
    pkgs.python310Packages.scikit-learn
    pkgs.python310Packages.requests
    pkgs.python310Packages.websocket-client
    pkgs.python310Packages.pylint
    pkgs.python310Packages.flake8
    pkgs.python310Packages.black
    pkgs.python310Packages.pytest
    pkgs.tk
    pkgs.tcl
  ];
}
