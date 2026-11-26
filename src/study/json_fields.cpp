#include "study/json_fields.hpp"

#include <algorithm>
#include <cmath>
#include <format>

namespace gso::study::detail {
namespace {

std::string typeName(const Json& value)
{
    return value.type_name();
}

// Levenshtein distance, used only to suggest the intended key when one is misspelled.
std::size_t editDistance(const std::string& a, const std::string& b)
{
    std::vector<std::size_t> previous(b.size() + 1);
    std::vector<std::size_t> current(b.size() + 1);
    for (std::size_t j = 0; j <= b.size(); ++j) {
        previous[j] = j;
    }
    for (std::size_t i = 1; i <= a.size(); ++i) {
        current[0] = i;
        for (std::size_t j = 1; j <= b.size(); ++j) {
            const std::size_t substitution = previous[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1);
            current[j] = std::min({previous[j] + 1, current[j - 1] + 1, substitution});
        }
        std::swap(previous, current);
    }
    return previous[b.size()];
}

}  // namespace

Fields::Fields(const Json& object, std::string path) : object_(object), path_(std::move(path))
{
    if (!object_.is_object()) {
        throw InputError(path_.empty() ? "(document)" : path_,
                         std::format("expected an object, found {}", typeName(object_)));
    }
}

Fields rootFields(const Json& document)
{
    return Fields(document, "");
}

std::string Fields::pathOf(const std::string& key) const
{
    return path_.empty() ? key : path_ + "." + key;
}

bool Fields::has(const std::string& key) const
{
    return object_.contains(key);
}

void Fields::fail(const std::string& key, const std::string& message) const
{
    throw InputError(key.empty() ? path_ : pathOf(key), message);
}

const Json& Fields::require(const std::string& key)
{
    const auto entry = object_.find(key);
    if (entry == object_.end()) {
        for (const auto& [present, value] : object_.items()) {
            if (editDistance(present, key) <= 2) {
                fail(key, std::format("is required (found '{}' instead; is it misspelled?)", present));
            }
        }
        fail(key, "is required");
    }
    read_.insert(key);
    return *entry;
}

const Json& Fields::raw(const std::string& key)
{
    return require(key);
}

double Fields::number(const std::string& key)
{
    const Json& value = require(key);
    if (!value.is_number()) {
        fail(key, std::format("expected a number, found {}", typeName(value)));
    }
    const double result = value.get<double>();
    if (!std::isfinite(result)) {
        fail(key, "must be finite");
    }
    return result;
}

int Fields::integer(const std::string& key)
{
    const Json& value = require(key);
    if (!value.is_number_integer()) {
        fail(key, std::format("expected an integer, found {}", typeName(value)));
    }
    return value.get<int>();
}

std::string Fields::text(const std::string& key)
{
    const Json& value = require(key);
    if (!value.is_string()) {
        fail(key, std::format("expected a string, found {}", typeName(value)));
    }
    return value.get<std::string>();
}

Eigen::Vector3d Fields::vector3(const std::string& key)
{
    const Json& value = require(key);
    if (!value.is_array() || value.size() != 3) {
        fail(key, "expected an array of three numbers");
    }
    Eigen::Vector3d result;
    for (int i = 0; i < 3; ++i) {
        const Json& component = value[static_cast<std::size_t>(i)];
        if (!component.is_number() || !std::isfinite(component.get<double>())) {
            fail(key, "expected an array of three finite numbers");
        }
        result[i] = component.get<double>();
    }
    return result;
}

std::vector<std::string> Fields::textList(const std::string& key)
{
    const Json& value = require(key);
    if (!value.is_array()) {
        fail(key, std::format("expected an array of strings, found {}", typeName(value)));
    }
    std::vector<std::string> result;
    for (const Json& item : value) {
        if (!item.is_string()) {
            fail(key, "expected an array of strings");
        }
        result.push_back(item.get<std::string>());
    }
    return result;
}

Fields Fields::object(const std::string& key)
{
    return Fields(require(key), pathOf(key));
}

std::vector<Fields> Fields::objectList(const std::string& key)
{
    const Json& value = require(key);
    if (!value.is_array()) {
        fail(key, std::format("expected an array, found {}", typeName(value)));
    }
    std::vector<Fields> result;
    for (std::size_t i = 0; i < value.size(); ++i) {
        result.emplace_back(value[i], std::format("{}[{}]", pathOf(key), i));
    }
    return result;
}

std::optional<double> Fields::optionalNumber(const std::string& key)
{
    if (!has(key)) {
        return std::nullopt;
    }
    return number(key);
}

std::optional<std::string> Fields::optionalText(const std::string& key)
{
    if (!has(key)) {
        return std::nullopt;
    }
    return text(key);
}

double Fields::positive(const std::string& key)
{
    const double value = number(key);
    if (value <= 0.0) {
        fail(key, std::format("must be positive, found {}", value));
    }
    return value;
}

double Fields::nonNegative(const std::string& key)
{
    const double value = number(key);
    if (value < 0.0) {
        fail(key, std::format("must not be negative, found {}", value));
    }
    return value;
}

std::pair<std::string, Fields> Fields::singleVariant(const std::string& key)
{
    const Json& value = require(key);
    if (!value.is_object() || value.size() != 1) {
        fail(key, "expected an object with exactly one key naming the kind, e.g. {\"isotropic\": {...}}");
    }
    const std::string kind = value.begin().key();
    return {kind, Fields(value.begin().value(), pathOf(key) + "." + kind)};
}

void Fields::finish() const
{
    for (const auto& [key, value] : object_.items()) {
        if (!read_.contains(key)) {
            fail(key, "is not a recognised key");
        }
    }
}

}  // namespace gso::study::detail
