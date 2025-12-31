/*
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2025 John William Creighton (s243a)
 *
 * Example: Distributed Processing Pipeline
 *
 * This example demonstrates a distributed data processing system:
 * - API Gateway (Go): Entry point, routing
 * - Transform Service (Python): Data transformation
 * - ML Service (Go): Machine learning inference
 * - Aggregator (Rust): High-performance aggregation
 *
 * Uses HTTP for request/response and sockets for streaming.
 */

:- use_module('../../src/unifyweaver/glue/network_glue').

%% ============================================
%% Service Definitions
%% ============================================

%% Register the services
register_services :-
    register_service(gateway, 'http://localhost:8080', [timeout(30)]),
    register_service(transform, 'http://localhost:8081', [timeout(60)]),
    register_service(ml_service, 'http://localhost:8082', [timeout(120)]),
    register_service(aggregator, 'http://localhost:8083', [timeout(60)]).

%% ============================================
%% API Gateway (Go)
%% ============================================

generate_gateway(Code) :-
    generate_go_http_server(
        [
            endpoint('/api/process', gateway_process, []),
            endpoint('/api/batch', gateway_batch, []),
            endpoint('/health', health_check, [])
        ],
        [port(8080), cors(true)],
        ServerCode
    ),

    % Add custom handler implementations
    format(atom(Code), '~w

// Handler implementations

func gateway_process(data interface{}) (interface{}, error) {
    // Route to transform service
    result, err := callTransformService(data)
    if err != nil {
        return nil, err
    }

    // Then to ML service
    return callMLService(result)
}

func gateway_batch(data interface{}) (interface{}, error) {
    items, ok := data.([]interface{})
    if !ok {
        return nil, fmt.Errorf("expected array input")
    }

    results := make([]interface{}, len(items))
    for i, item := range items {
        result, err := gateway_process(item)
        if err != nil {
            results[i] = map[string]interface{}{"error": err.Error()}
        } else {
            results[i] = result
        }
    }
    return results, nil
}

func health_check(data interface{}) (interface{}, error) {
    return map[string]interface{}{
        "status": "healthy",
        "services": map[string]string{
            "transform": "http://localhost:8081",
            "ml": "http://localhost:8082",
            "aggregator": "http://localhost:8083",
        },
    }, nil
}

// Service clients
func callTransformService(data interface{}) (interface{}, error) {
    return callRemoteService("http://localhost:8081/transform", data)
}

func callMLService(data interface{}) (interface{}, error) {
    return callRemoteService("http://localhost:8082/predict", data)
}

func callRemoteService(url string, data interface{}) (interface{}, error) {
    reqBody, _ := json.Marshal(Request{Data: data})
    resp, err := http.Post(url, "application/json", bytes.NewReader(reqBody))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)
    var result Response
    json.Unmarshal(body, &result)

    if !result.Success {
        return nil, fmt.Errorf(result.Error)
    }
    return result.Data, nil
}

var _ = fmt.Sprintf  // suppress unused import
', [ServerCode]).

%% ============================================
%% Transform Service (Python)
%% ============================================

generate_transform_service(Code) :-
    generate_python_http_server(
        [
            endpoint('/transform', transform, [methods(['POST'])]),
            endpoint('/batch', batch_transform, [methods(['POST'])]),
            endpoint('/health', health, [methods(['GET'])])
        ],
        [port(8081), cors(true)],
        ServerCode
    ),

    format(atom(Code), '~w

# Handler implementations

def normalize(value, min_val=0, max_val=100):
    """Normalize value to 0-1 range."""
    return max(0, min(1, (value - min_val) / (max_val - min_val)))

def transform(data):
    """Transform a single record."""
    if data is None:
        return {"error": "No data provided"}

    result = dict(data) if isinstance(data, dict) else {"raw": data}

    # Add computed fields
    if "value" in result:
        result["normalized"] = normalize(float(result["value"]))

    if "timestamp" in result:
        from datetime import datetime
        try:
            dt = datetime.fromisoformat(result["timestamp"].replace("Z", "+00:00"))
            result["hour"] = dt.hour
            result["day_of_week"] = dt.weekday()
        except:
            pass

    result["transformed"] = True
    return result

def batch_transform(data):
    """Transform multiple records."""
    if not isinstance(data, list):
        return {"error": "Expected list input"}

    return [transform(item) for item in data]

def health(data):
    """Health check."""
    return {"status": "healthy", "service": "transform"}
', [ServerCode]).

%% ============================================
%% ML Service (Go)
%% ============================================

generate_ml_service(Code) :-
    generate_go_http_server(
        [
            endpoint('/predict', predict, []),
            endpoint('/classify', classify, []),
            endpoint('/health', ml_health, [])
        ],
        [port(8082), cors(true)],
        ServerCode
    ),

    format(atom(Code), '~w

import (
    "math"
    "math/rand"
)

// Simple ML "models" (in production, load real models)

func predict(data interface{}) (interface{}, error) {
    record, ok := data.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("expected object input")
    }

    // Simple prediction based on normalized value
    normalized, _ := record["normalized"].(float64)
    if normalized == 0 {
        normalized = 0.5
    }

    // "Model" prediction
    score := normalized * 0.7 + rand.Float64() * 0.3
    confidence := 0.8 + rand.Float64() * 0.2

    record["prediction"] = map[string]interface{}{
        "score": math.Round(score * 1000) / 1000,
        "confidence": math.Round(confidence * 1000) / 1000,
        "label": scoreToLabel(score),
    }

    return record, nil
}

func scoreToLabel(score float64) string {
    if score > 0.7 {
        return "high"
    } else if score > 0.4 {
        return "medium"
    }
    return "low"
}

func classify(data interface{}) (interface{}, error) {
    record, ok := data.(map[string]interface{})
    if !ok {
        return nil, fmt.Errorf("expected object input")
    }

    // Simple classification
    categories := []string{"A", "B", "C", "D"}
    idx := rand.Intn(len(categories))

    record["classification"] = map[string]interface{}{
        "category": categories[idx],
        "probabilities": map[string]float64{
            "A": 0.25,
            "B": 0.25,
            "C": 0.25,
            "D": 0.25,
        },
    }

    return record, nil
}

func ml_health(data interface{}) (interface{}, error) {
    return map[string]interface{}{
        "status": "healthy",
        "service": "ml",
        "models": []string{"predictor", "classifier"},
    }, nil
}

var _ = math.Round  // suppress unused import
', [ServerCode]).

%% ============================================
%% Aggregator Service (Rust)
%% ============================================

generate_aggregator_service(Code) :-
    generate_rust_http_server(
        [
            endpoint('/aggregate', aggregate, []),
            endpoint('/stats', compute_stats, []),
            endpoint('/health', rust_health, [])
        ],
        [port(8083)],
        ServerCode
    ),

    format(atom(Code), '~w

// Handler implementations

fn aggregate(data: Option<serde_json::Value>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let items = data
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    let mut sum = 0.0;
    let mut count = 0;

    for item in &items {
        if let Some(score) = item
            .get("prediction")
            .and_then(|p| p.get("score"))
            .and_then(|s| s.as_f64())
        {
            sum += score;
            count += 1;
        }
    }

    let avg = if count > 0 { sum / count as f64 } else { 0.0 };

    Ok(serde_json::json!({
        "total": items.len(),
        "scored": count,
        "average_score": (avg * 1000.0).round() / 1000.0,
        "aggregated": true
    }))
}

fn compute_stats(data: Option<serde_json::Value>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let items = data
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    let scores: Vec<f64> = items
        .iter()
        .filter_map(|item| {
            item.get("prediction")
                .and_then(|p| p.get("score"))
                .and_then(|s| s.as_f64())
        })
        .collect();

    let count = scores.len();
    if count == 0 {
        return Ok(serde_json::json!({"error": "No scores found"}));
    }

    let sum: f64 = scores.iter().sum();
    let mean = sum / count as f64;
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance: f64 = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();

    Ok(serde_json::json!({
        "count": count,
        "sum": (sum * 1000.0).round() / 1000.0,
        "mean": (mean * 1000.0).round() / 1000.0,
        "min": (min * 1000.0).round() / 1000.0,
        "max": (max * 1000.0).round() / 1000.0,
        "std_dev": (std_dev * 1000.0).round() / 1000.0
    }))
}

fn rust_health(_data: Option<serde_json::Value>) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    Ok(serde_json::json!({
        "status": "healthy",
        "service": "aggregator"
    }))
}
', [ServerCode]).

%% ============================================
%% Client Code
%% ============================================

generate_client(Code) :-
    generate_python_http_client(
        [
            service_def(gateway, 'http://localhost:8080', ['/api/process', '/api/batch', '/health'])
        ],
        [timeout(60)],
        Code
    ).

%% ============================================
%% Docker Compose
%% ============================================

generate_docker_compose(Compose) :-
    Compose = 'version: "3.8"

services:
  gateway:
    build:
      context: .
      dockerfile: Dockerfile.gateway
    ports:
      - "8080:8080"
    depends_on:
      - transform
      - ml
      - aggregator

  transform:
    build:
      context: .
      dockerfile: Dockerfile.transform
    ports:
      - "8081:8081"

  ml:
    build:
      context: .
      dockerfile: Dockerfile.ml
    ports:
      - "8082:8082"

  aggregator:
    build:
      context: .
      dockerfile: Dockerfile.aggregator
    ports:
      - "8083:8083"
'.

%% ============================================
%% Main: Generate All Files
%% ============================================

generate_all :-
    format('Generating distributed pipeline...~n~n'),

    % Register services
    register_services,

    % Gateway (Go)
    generate_gateway(GatewayCode),
    open('gateway.go', write, S1),
    write(S1, GatewayCode),
    close(S1),
    format('  Created: gateway.go~n'),

    % Transform (Python)
    generate_transform_service(TransformCode),
    open('transform_service.py', write, S2),
    write(S2, TransformCode),
    close(S2),
    format('  Created: transform_service.py~n'),

    % ML (Go)
    generate_ml_service(MLCode),
    open('ml_service.go', write, S3),
    write(S3, MLCode),
    close(S3),
    format('  Created: ml_service.go~n'),

    % Aggregator (Rust)
    generate_aggregator_service(AggregatorCode),
    open('aggregator_service.rs', write, S4),
    write(S4, AggregatorCode),
    close(S4),
    format('  Created: aggregator_service.rs~n'),

    % Client
    generate_client(ClientCode),
    open('client.py', write, S5),
    write(S5, ClientCode),
    close(S5),
    format('  Created: client.py~n'),

    % Docker Compose
    generate_docker_compose(ComposeCode),
    open('docker-compose.yml', write, S6),
    write(S6, ComposeCode),
    close(S6),
    format('  Created: docker-compose.yml~n'),

    format('~nDone! Start services with:~n'),
    format('  docker-compose up~n'),
    format('Or run locally:~n'),
    format('  go run gateway.go &~n'),
    format('  python transform_service.py &~n'),
    format('  go run ml_service.go &~n').

%% ============================================
%% Main
%% ============================================

:- initialization(generate_all, main).
