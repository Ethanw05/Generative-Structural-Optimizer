#pragma once

// Strict access to JSON objects for the study reader. Every value is read through a
// Fields object that remembers its path in the document ("cases[2].loads[0]") for error
// messages, and records which keys were read so finish() can reject unknown ones.

#include "gso/core/error.hpp"

#include <Eigen/Core>
#include <nlohmann/json.hpp>

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace gso::study::detail {

using Json = nlohmann::json;

class Fields {
public:
    Fields(const Json& object, std::string path);

    const std::string& path() const { return path_; }
    std::string pathOf(const std::string& key) const;
    bool has(const std::string& key) const;

    // Required values: an InputError if missing or of the wrong type.
    double number(const std::string& key);
    int integer(const std::string& key);
    std::string text(const std::string& key);
    Eigen::Vector3d vector3(const std::string& key);
    std::vector<std::string> textList(const std::string& key);
    Fields object(const std::string& key);
    std::vector<Fields> objectList(const std::string& key);
    // The value of `key` as raw JSON, for values with several allowed forms.
    const Json& raw(const std::string& key);

    std::optional<double> optionalNumber(const std::string& key);
    std::optional<std::string> optionalText(const std::string& key);

    // Constraints on values that were read.
    double positive(const std::string& key);
    double nonNegative(const std::string& key);

    // An object with exactly one key, used for tagged alternatives such as
    // {"isotropic": {...}}. Returns the key and its object.
    std::pair<std::string, Fields> singleVariant(const std::string& key);

    // Throws InputError for every key that was not read.
    void finish() const;

    [[noreturn]] void fail(const std::string& key, const std::string& message) const;

private:
    const Json& require(const std::string& key);

    const Json& object_;
    std::string path_;
    std::set<std::string> read_;
};

// Reads a whole document root.
Fields rootFields(const Json& document);

}  // namespace gso::study::detail
