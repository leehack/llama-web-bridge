// Log level names and thresholds.

export function logLevelForName(level) {
  switch (level) {
    case 'debug':
      return 0;
    case 'log':
    case 'info':
      return 1;
    case 'warn':
      return 2;
    case 'error':
      return 3;
    default:
      return 1;
  }
}

export function logThresholdForConfiguredLevel(level) {
  switch (level) {
    case 0:
      return 99;
    case 1:
      return 0;
    case 2:
      return 1;
    case 3:
      return 2;
    case 4:
      return 3;
    default:
      return 1;
  }
}
