module.exports = {
  root: true,
  ignorePatterns: [
    'venv/',
    'htmlcov/',
    'coverage/',
    '**/site-packages/',
    '**/sphinx/',
    'dashboard-new/dist/',
    'node_modules/',
    '*.log',
  ],
  extends: [
    'eslint:recommended',
    'plugin:react/recommended',
    'plugin:@typescript-eslint/recommended',
  ],
  parser: '@typescript-eslint/parser',
  plugins: ['react', '@typescript-eslint'],
  rules: {
    // Twoje reguły
  },
};
