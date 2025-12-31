#!/usr/bin/env ruby
# frozen_string_literal: true

# PyCall.rb + RPyC Example
#
# Demonstrates using PyCall.rb to embed Python in Ruby and access RPyC
# for remote Python computation.
#
# Usage:
#   1. Install: gem install pycall
#   2. Start RPyC server: python examples/rpyc-integration/rpyc_server.py
#   3. Run: ruby rpyc_client.rb

require 'pycall'

puts "PyCall.rb + RPyC Integration"
puts "============================="
puts

begin
  # Import rpyc module
  rpyc = PyCall.import_module('rpyc')

  puts "Connecting to RPyC server..."

  # Connect to server
  conn = rpyc.classic.connect('localhost', 18812)

  # Test 1: math.sqrt
  puts
  puts "Test 1: Remote math.sqrt"
  math = conn.modules.math
  result = math.sqrt(16).to_f  # Convert Python float to Ruby float
  puts "  math.sqrt(16) = #{result}"
  raise "Expected 4.0" unless (result - 4.0).abs < 0.001
  puts "  ✓ Passed"

  # Test 2: numpy.mean
  puts
  puts "Test 2: Remote numpy.mean"
  np = conn.modules.numpy
  arr = np.array([1, 2, 3, 4, 5])
  mean = np.mean(arr).to_f  # Convert Python float to Ruby float
  puts "  numpy.mean([1,2,3,4,5]) = #{mean}"
  raise "Expected 3.0" unless (mean - 3.0).abs < 0.001
  puts "  ✓ Passed"

  # Test 3: Get server Python version
  puts
  puts "Test 3: Server Python version"
  sys = conn.modules.sys
  version = sys.version.to_s.split.first
  puts "  Server Python: #{version}"
  puts "  ✓ Passed"

  # Close connection
  conn.close

  puts
  puts "============================="
  puts "All tests passed!"

rescue LoadError => e
  puts "Error: #{e.message}"
  puts
  puts "Install PyCall.rb:"
  puts "  gem install pycall"

rescue PyCall::PyError => e
  puts "Python error: #{e.message}"
  puts
  puts "Make sure rpyc is installed:"
  puts "  pip install rpyc"

rescue Errno::ECONNREFUSED
  puts "Connection refused"
  puts
  puts "Make sure RPyC server is running:"
  puts "  python examples/rpyc-integration/rpyc_server.py"

rescue => e
  puts "Error: #{e.message}"
  puts e.backtrace.first(5).join("\n")
end
