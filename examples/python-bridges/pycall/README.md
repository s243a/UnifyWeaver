# PyCall.rb + RPyC Example

Use PyCall.rb to embed CPython in Ruby and access RPyC.

## Status

| Feature | Status |
|---------|--------|
| PyCall.rb binding | ✅ Active, v1.5.2 |
| RPyC access | ✅ Tested and working |

## Overview

[PyCall.rb](https://github.com/mrkn/pycall.rb) allows Ruby programs to call
Python functions and use Python objects. It embeds real CPython, enabling
full access to RPyC's live object proxies.

## Prerequisites

### 1. Ruby 2.7+

```bash
ruby --version  # Should be 2.7+
```

### 2. PyCall.rb gem

```bash
gem install pycall
```

### 3. Python 3.8+ with rpyc

```bash
pip install rpyc
```

### 4. RPyC Server Running

```bash
python examples/rpyc-integration/rpyc_server.py
```

## Usage

### Simple Script

```ruby
#!/usr/bin/env ruby
# rpyc_client.rb

require 'pycall'

# Import rpyc
rpyc = PyCall.import_module('rpyc')

# Connect to server
conn = rpyc.classic.connect('localhost', 18812)

# Use remote math module
math = conn.modules.math
result = math.sqrt(16)
puts "math.sqrt(16) = #{result}"  # 4.0

# Use remote numpy
np = conn.modules.numpy
arr = np.array([1, 2, 3, 4, 5])
mean = np.mean(arr)
puts "numpy.mean([1,2,3,4,5]) = #{mean}"  # 3.0

# Close connection
conn.close()
```

### Running

```bash
ruby rpyc_client.rb
```

## Expected Output

```
math.sqrt(16) = 4.0
numpy.mean([1,2,3,4,5]) = 3.0
```

## Key Concepts

### Importing Python Modules

```ruby
require 'pycall'

# Import a module
numpy = PyCall.import_module('numpy')

# Import with alias
np = PyCall.import_module('numpy')
```

### Calling Python Functions

```ruby
# Call function
result = np.sqrt(16)

# Call method
arr = np.array([1, 2, 3])
mean = np.mean(arr)
```

### Attribute Access

```ruby
# Access attribute
version = conn.modules.sys.version
```

### Type Conversion

PyCall.rb automatically converts between Ruby and Python types:

| Ruby | Python |
|------|--------|
| Integer | int |
| Float | float |
| String | str |
| Array | list |
| Hash | dict |
| true/false | True/False |
| nil | None |

**Note:** Remote Python objects (via RPyC) may need explicit conversion:
```ruby
# Python float from RPyC needs .to_f for Ruby numeric operations
result = math.sqrt(16).to_f
(result - 4.0).abs < 0.001  # Now works with Ruby's abs method
```

## Advanced: Ruby Class Wrapper

```ruby
require 'pycall'

class RPyCClient
  def initialize(host = 'localhost', port = 18812)
    @rpyc = PyCall.import_module('rpyc')
    @conn = @rpyc.classic.connect(host, port)
  end

  def sqrt(value)
    @conn.modules.math.sqrt(value)
  end

  def numpy_mean(values)
    np = @conn.modules.numpy
    arr = np.array(values)
    np.mean(arr)
  end

  def python_version
    @conn.modules.sys.version.split.first
  end

  def close
    @conn.close
  end
end

# Usage
client = RPyCClient.new
puts client.sqrt(16)           # 4.0
puts client.numpy_mean([1,2,3,4,5])  # 3.0
puts client.python_version     # 3.8.10
client.close
```

## Error Handling

```ruby
begin
  rpyc = PyCall.import_module('rpyc')
  conn = rpyc.classic.connect('localhost', 18812)
rescue PyCall::PyError => e
  puts "Python error: #{e.message}"
rescue Errno::ECONNREFUSED
  puts "Connection refused - is RPyC server running?"
end
```

## Gemfile Setup

For project use:

```ruby
# Gemfile
source 'https://rubygems.org'

gem 'pycall', '~> 1.5'
```

Then:

```bash
bundle install
bundle exec ruby rpyc_client.rb
```

## PyCall.rb vs Other Bridges

| Aspect | PyCall.rb (Ruby) | JPype (Java) | PyO3 (Rust) |
|--------|------------------|--------------|-------------|
| Dynamic Typing | Yes | Yes | No |
| Syntax | Ruby-like | Java-like | Rust-like |
| Object Conversion | Automatic | Manual | Manual |
| Active Development | Yes | Yes | Very Active |

## Troubleshooting

### "LoadError: cannot load pycall"

Install the gem:
```bash
gem install pycall
```

### "PyCall::PyError: No module named 'rpyc'"

Install rpyc in your Python environment:
```bash
pip install rpyc
```

### "Errno::ECONNREFUSED"

Start the RPyC server:
```bash
python examples/rpyc-integration/rpyc_server.py
```

## Resources

- [PyCall.rb GitHub](https://github.com/mrkn/pycall.rb)
- [PyCall.rb RubyDoc](https://www.rubydoc.info/gems/pycall)
