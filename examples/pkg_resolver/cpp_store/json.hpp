#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace json {

enum class Type { Null, Bool, Int, String, Array, Object };

class Json {
public:
    Json();  // default-constructs Null
    static Json makeNull();
    static Json makeBool(bool b);
    static Json makeInt(int64_t n);
    static Json makeString(std::string s);
    static Json makeArray();
    static Json makeObject();

    Type type() const;
    bool isNull() const; bool isBool() const; bool isInt() const;
    bool isString() const; bool isArray() const; bool isObject() const;

    bool                asBool()   const;
    int64_t             asInt()    const;
    const std::string&  asString() const;
    const std::vector<Json>& asArray() const;
    std::vector<Json>&       asArray();
    const std::vector<std::pair<std::string, Json>>& asObject() const;
    std::vector<std::pair<std::string, Json>>&       asObject();

    bool has(const std::string& key) const;
    const Json* find(const std::string& key) const;  // nullptr if absent
    void set(const std::string& key, Json v);         // replace-in-place if present, else append

    void push_back(Json v);
    size_t size() const;                       // Array or Object element count; else 0
    const Json& operator[](size_t i) const;    // Array indexing

private:
    Type type_{Type::Null};
    bool bool_{false};
    int64_t int_{0};
    std::string str_;
    std::vector<Json> arr_;
    std::vector<std::pair<std::string, Json>> obj_;
};

class ParseError : public std::runtime_error {
public:
    explicit ParseError(const std::string& msg) : std::runtime_error(msg) {}
};

Json parse(const std::string& text);   // throws ParseError on malformed input / trailing junk
std::string dump(const Json& v);       // COMPACT output: no whitespace between tokens

}  // namespace json
