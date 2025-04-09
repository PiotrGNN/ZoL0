
{ pkgs }: {
  deps = [
    pkgs.libyaml
    pkgs.gmp
    pkgs.glibcLocales
    pkgs.tk
    pkgs.tcl
    pkgs.qhull
    pkgs.pkg-config
    pkgs.gtk3
    pkgs.gobject-introspection
    pkgs.ghostscript
    pkgs.freetype
    pkgs.ffmpeg-full
    pkgs.cairo
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.numpy
    pkgs.python310Packages.pandas
    pkgs.python310Packages.scikit-learn
    pkgs.python310Packages.flask
    pkgs.python310Packages.matplotlib
    pkgs.python310Packages.setuptools
    pkgs.python310Packages.wheel
  ];
}
