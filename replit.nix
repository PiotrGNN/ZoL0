
{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.flask
    pkgs.python310Packages.numpy
    pkgs.python310Packages.pandas
    pkgs.python310Packages.scikit-learn
    pkgs.python310Packages.matplotlib
    pkgs.gcc
  ];
}
