package utils

import "strings"

// GetLocal retrieves a value from a map ignoring namespace prefixes.
// It tries: exact match, @prefix, and any namespace:localname pattern.
// Examples: "title", "@title", "dcterms:title", "@dcterms:title"
func GetLocal(data map[string]interface{}, localName string) interface{} {
	// Try exact match
	if val, ok := data[localName]; ok {
		return val
	}

	// Try @prefix
	atKey := "@" + localName
	if val, ok := data[atKey]; ok {
		return val
	}

	// Try any namespace:localname pattern
	suffix := ":" + localName
	for key, val := range data {
		if strings.HasSuffix(key, suffix) {
			return val
		}
	}

	return nil
}

// GetLocalString is a convenience wrapper that returns a string or empty string.
func GetLocalString(data map[string]interface{}, localName string) string {
	val := GetLocal(data, localName)
	if val == nil {
		return ""
	}
	if s, ok := val.(string); ok {
		return s
	}
	// Handle nested map case (e.g., child element with text content)
	if m, ok := val.(map[string]interface{}); ok {
		if text, ok := m["text"].(string); ok {
			return text
		}
	}
	return ""
}
