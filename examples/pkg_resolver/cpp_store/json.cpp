#include "json.hpp"

#include <cctype>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace json {

Json::Json() : type_(Type::Null), bool_(false), int_(0), str_(), arr_(), obj_() {}

Json Json::makeNull() {
    return Json();
}

Json Json::makeBool(bool b) {
    Json j;
    j.type_ = Type::Bool;
    j.bool_ = b;
    return j;
}

Json Json::makeInt(int64_t n) {
    Json j;
    j.type_ = Type::Int;
    j.int_ = n;
    return j;
}

Json Json::makeString(std::string s) {
    Json j;
    j.type_ = Type::String;
    j.str_ = std::move(s);
    return j;
}

Json Json::makeArray() {
    Json j;
    j.type_ = Type::Array;
    return j;
}

Json Json::makeObject() {
    Json j;
    j.type_ = Type::Object;
    return j;
}

Type Json::type() const { return type_; }
bool Json::isNull() const { return type_ == Type::Null; }
bool Json::isBool() const { return type_ == Type::Bool; }
bool Json::isInt() const { return type_ == Type::Int; }
bool Json::isString() const { return type_ == Type::String; }
bool Json::isArray() const { return type_ == Type::Array; }
bool Json::isObject() const { return type_ == Type::Object; }

bool Json::asBool() const {
    if (type_ != Type::Bool) {
        throw std::runtime_error("Json: value is not a bool");
    }
    return bool_;
}

int64_t Json::asInt() const {
    if (type_ != Type::Int) {
        throw std::runtime_error("Json: value is not an int");
    }
    return int_;
}

const std::string& Json::asString() const {
    if (type_ != Type::String) {
        throw std::runtime_error("Json: value is not a string");
    }
    return str_;
}

const std::vector<Json>& Json::asArray() const {
    if (type_ != Type::Array) {
        throw std::runtime_error("Json: value is not an array");
    }
    return arr_;
}

std::vector<Json>& Json::asArray() {
    if (type_ != Type::Array) {
        throw std::runtime_error("Json: value is not an array");
    }
    return arr_;
}

const std::vector<std::pair<std::string, Json>>& Json::asObject() const {
    if (type_ != Type::Object) {
        throw std::runtime_error("Json: value is not an object");
    }
    return obj_;
}

std::vector<std::pair<std::string, Json>>& Json::asObject() {
    if (type_ != Type::Object) {
        throw std::runtime_error("Json: value is not an object");
    }
    return obj_;
}

bool Json::has(const std::string& key) const {
    return find(key) != nullptr;
}

const Json* Json::find(const std::string& key) const {
    if (type_ != Type::Object) {
        return nullptr;
    }
    for (const auto& kv : obj_) {
        if (kv.first == key) {
            return &kv.second;
        }
    }
    return nullptr;
}

void Json::set(const std::string& key, Json v) {
    if (type_ != Type::Object) {
        throw std::runtime_error("Json::set called on non-object");
    }
    for (auto& kv : obj_) {
        if (kv.first == key) {
            kv.second = std::move(v);
            return;
        }
    }
    obj_.emplace_back(key, std::move(v));
}

void Json::push_back(Json v) {
    if (type_ != Type::Array) {
        throw std::runtime_error("Json::push_back called on non-array");
    }
    arr_.push_back(std::move(v));
}

size_t Json::size() const {
    if (type_ == Type::Array) {
        return arr_.size();
    }
    if (type_ == Type::Object) {
        return obj_.size();
    }
    return 0;
}

const Json& Json::operator[](size_t i) const {
    if (type_ != Type::Array) {
        throw std::runtime_error("Json::operator[] called on non-array");
    }
    if (i >= arr_.size()) {
        throw std::out_of_range("Json::operator[] index out of range");
    }
    return arr_[i];
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text), pos_(0) {}

    void skipWhitespace() {
        while (pos_ < text_.size()) {
            char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                pos_++;
            } else {
                break;
            }
        }
    }

    bool eof() const {
        return pos_ >= text_.size();
    }

    Json parseValue() {
        skipWhitespace();
        if (eof()) {
            throw ParseError("Unexpected end of input");
        }
        char c = text_[pos_];
        if (c == '{') {
            return parseObject();
        } else if (c == '[') {
            return parseArray();
        } else if (c == '"') {
            return Json::makeString(parseRawString());
        } else if (c == 't') {
            return parseTrue();
        } else if (c == 'f') {
            return parseFalse();
        } else if (c == 'n') {
            return parseNull();
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            return parseNumber();
        } else {
            throw ParseError(std::string("Unexpected character: ") + c);
        }
    }

private:
    const std::string& text_;
    size_t pos_;

    Json parseTrue() {
        if (pos_ + 4 <= text_.size() && text_.compare(pos_, 4, "true") == 0) {
            pos_ += 4;
            return Json::makeBool(true);
        }
        throw ParseError("Expected 'true'");
    }

    Json parseFalse() {
        if (pos_ + 5 <= text_.size() && text_.compare(pos_, 5, "false") == 0) {
            pos_ += 5;
            return Json::makeBool(false);
        }
        throw ParseError("Expected 'false'");
    }

    Json parseNull() {
        if (pos_ + 4 <= text_.size() && text_.compare(pos_, 4, "null") == 0) {
            pos_ += 4;
            return Json::makeNull();
        }
        throw ParseError("Expected 'null'");
    }

    std::string parseRawString() {
        pos_++;  // consume opening quote
        std::string s;
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"') {
                return s;
            }
            if (c == '\\') {
                if (pos_ >= text_.size()) {
                    throw ParseError("Unfinished escape sequence in string");
                }
                char esc = text_[pos_++];
                switch (esc) {
                    case '"':  s.push_back('"'); break;
                    case '\\': s.push_back('\\'); break;
                    case '/':  s.push_back('/'); break;
                    case 'b':  s.push_back('\b'); break;
                    case 'f':  s.push_back('\f'); break;
                    case 'n':  s.push_back('\n'); break;
                    case 'r':  s.push_back('\r'); break;
                    case 't':  s.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) {
                            throw ParseError("Incomplete \\u hex escape in string");
                        }
                        uint32_t cp = 0;
                        for (int i = 0; i < 4; ++i) {
                            char h = text_[pos_++];
                            cp <<= 4;
                            if (h >= '0' && h <= '9') cp |= static_cast<uint32_t>(h - '0');
                            else if (h >= 'a' && h <= 'f') cp |= static_cast<uint32_t>(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') cp |= static_cast<uint32_t>(h - 'A' + 10);
                            else throw ParseError("Invalid hex digit in \\u escape");
                        }
                        if (cp <= 0x7F) {
                            s.push_back(static_cast<char>(cp));
                        } else if (cp <= 0x7FF) {
                            s.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
                            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        } else {
                            s.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
                            s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                            s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                        }
                        break;
                    }
                    default:
                        throw ParseError(std::string("Invalid escape character: \\") + esc);
                }
            } else {
                if (static_cast<unsigned char>(c) < 0x20) {
                    throw ParseError("Unescaped control character in string");
                }
                s.push_back(c);
            }
        }
        throw ParseError("Unterminated string");
    }

    Json parseNumber() {
        bool negative = false;
        if (text_[pos_] == '-') {
            negative = true;
            pos_++;
            if (pos_ >= text_.size() || text_[pos_] < '0' || text_[pos_] > '9') {
                throw ParseError("Expected digit after '-'");
            }
        }

        uint64_t val = 0;
        if (text_[pos_] == '0') {
            pos_++;
            if (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                throw ParseError("Leading zeroes are not permitted in numbers");
            }
        } else {
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
                val = val * 10 + static_cast<uint64_t>(text_[pos_] - '0');
                pos_++;
            }
        }

        if (pos_ < text_.size() && (text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E')) {
            throw ParseError("Floating-point numbers not supported");
        }

        int64_t result = negative ? -static_cast<int64_t>(val) : static_cast<int64_t>(val);
        return Json::makeInt(result);
    }

    Json parseArray() {
        pos_++;  // consume '['
        Json arr = Json::makeArray();
        skipWhitespace();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            pos_++;
            return arr;
        }

        while (true) {
            Json elem = parseValue();
            arr.push_back(std::move(elem));
            skipWhitespace();
            if (eof()) {
                throw ParseError("Unterminated array");
            }
            if (text_[pos_] == ']') {
                pos_++;
                return arr;
            }
            if (text_[pos_] == ',') {
                pos_++;
                skipWhitespace();
                if (pos_ < text_.size() && text_[pos_] == ']') {
                    throw ParseError("Trailing comma in array");
                }
            } else {
                throw ParseError("Expected ',' or ']' in array");
            }
        }
    }

    Json parseObject() {
        pos_++;  // consume '{'
        Json obj = Json::makeObject();
        skipWhitespace();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            pos_++;
            return obj;
        }

        while (true) {
            skipWhitespace();
            if (eof() || text_[pos_] != '"') {
                throw ParseError("Expected string key in object");
            }
            std::string key = parseRawString();
            skipWhitespace();
            if (eof() || text_[pos_] != ':') {
                throw ParseError("Expected ':' after object key");
            }
            pos_++;  // consume ':'

            Json val = parseValue();
            obj.set(key, std::move(val));

            skipWhitespace();
            if (eof()) {
                throw ParseError("Unterminated object");
            }
            if (text_[pos_] == '}') {
                pos_++;
                return obj;
            }
            if (text_[pos_] == ',') {
                pos_++;
                skipWhitespace();
                if (pos_ < text_.size() && text_[pos_] == '}') {
                    throw ParseError("Trailing comma in object");
                }
            } else {
                throw ParseError("Expected ',' or '}' in object");
            }
        }
    }
};

void dumpString(const std::string& s, std::string& out) {
    out += '"';
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    out += '"';
}

void dumpValue(const Json& v, std::string& out) {
    switch (v.type()) {
        case Type::Null:
            out += "null";
            break;
        case Type::Bool:
            out += (v.asBool() ? "true" : "false");
            break;
        case Type::Int:
            out += std::to_string(v.asInt());
            break;
        case Type::String:
            dumpString(v.asString(), out);
            break;
        case Type::Array: {
            out += '[';
            const auto& arr = v.asArray();
            for (size_t i = 0; i < arr.size(); ++i) {
                if (i > 0) out += ',';
                dumpValue(arr[i], out);
            }
            out += ']';
            break;
        }
        case Type::Object: {
            out += '{';
            const auto& obj = v.asObject();
            for (size_t i = 0; i < obj.size(); ++i) {
                if (i > 0) out += ',';
                dumpString(obj[i].first, out);
                out += ':';
                dumpValue(obj[i].second, out);
            }
            out += '}';
            break;
        }
    }
}

}  // namespace

Json parse(const std::string& text) {
    Parser parser(text);
    parser.skipWhitespace();
    if (parser.eof()) {
        throw ParseError("Empty input");
    }
    Json val = parser.parseValue();
    parser.skipWhitespace();
    if (!parser.eof()) {
        throw ParseError("Trailing junk after JSON value");
    }
    return val;
}

std::string dump(const Json& v) {
    std::string out;
    dumpValue(v, out);
    return out;
}

}  // namespace json
