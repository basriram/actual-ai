async function suppressConsoleLogsAsync(callback: () => Promise<void>) {
  const originalConsoleLog = console.log;
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  console.log = () => {}; // Empty arrow function
  try {
    return await callback();
  } finally {
    // Restore even when the callback throws: a failed bank sync must not
    // silence every later log line for the rest of the process.
    console.log = originalConsoleLog;
  }
}

export default suppressConsoleLogsAsync;
