module.exports = {
  extends: [
    'stylelint-config-standard',
    'stylelint-config-recommended',
  ],
  ignoreFiles: [
    '**/venv/**/*.css',
    '**/htmlcov/**/*.css',
    '**/coverage/**/*.css',
    '**/site-packages/**/*.css',
    '**/sphinx/**/*.css',
    'dashboard-new/dist/**/*.css',
    'node_modules/**/*.css',
  ],
  rules: {
    // Twoje reguły
  },
};
