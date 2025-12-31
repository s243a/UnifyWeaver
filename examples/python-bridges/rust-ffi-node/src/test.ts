/**
 * Simple test script for the Rust FFI RPyC bridge.
 *
 * Usage: npm run test
 */

import { RpycBridge } from './rpyc_bridge';

async function main() {
  console.log('Node.js + Rust FFI + RPyC Integration Test');
  console.log('==========================================\n');

  const bridge = new RpycBridge();

  try {
    // Connect
    console.log('Connecting to RPyC server...');
    bridge.connect('localhost', 18812);
    console.log('  Connected!\n');

    // Test 1: math.sqrt
    console.log('Test 1: math.sqrt(16)');
    const sqrt = bridge.call<number>('math', 'sqrt', [16]);
    console.log(`  Result: ${sqrt}`);
    if (sqrt === 4) {
      console.log('  PASSED!\n');
    } else {
      console.log('  FAILED!\n');
      process.exit(1);
    }

    // Test 2: numpy.mean
    console.log('Test 2: numpy.mean([1, 2, 3, 4, 5])');
    const mean = bridge.call<number>('numpy', 'mean', [[1, 2, 3, 4, 5]]);
    console.log(`  Result: ${mean}`);
    if (mean === 3) {
      console.log('  PASSED!\n');
    } else {
      console.log('  FAILED!\n');
      process.exit(1);
    }

    // Test 3: math.pi
    console.log('Test 3: math.pi');
    const pi = bridge.getattr<number>('math', 'pi');
    console.log(`  Result: ${pi}`);
    if (pi > 3.14 && pi < 3.15) {
      console.log('  PASSED!\n');
    } else {
      console.log('  FAILED!\n');
      process.exit(1);
    }

    console.log('==========================================');
    console.log('All tests passed!');

  } catch (error) {
    console.error('Error:', error);
    process.exit(1);
  } finally {
    bridge.disconnect();
    console.log('\nConnection closed.');
  }
}

main();
