"""
Test data samples for comprehensive Model1 testing.
Includes multilingual inputs, edge cases, and performance test data.
"""

# Multilingual Test Samples
MULTILINGUAL_SAMPLES = {
    "english": [
        {
            "text": "Tsunami warning issued for Chennai coast. Evacuate immediately.",
            "expected_hazard": True,
            "expected_type": "tsunami",
            "expected_location": "Chennai",
            "urgency": "high"
        },
        {
            "text": "High waves reported at Marina Beach. Stay away from the water.",
            "expected_hazard": True,
            "expected_type": "high_waves",
            "expected_location": "Marina Beach",
            "urgency": "medium"
        },
        {
            "text": "Beautiful sunset at Kovalam Beach today. Perfect weather.",
            "expected_hazard": False,
            "expected_type": "none",
            "expected_location": "Kovalam Beach",
            "urgency": "low"
        }
    ],
    
    "hindi": [
        {
            "text": "चेन्नई के तट पर सुनामी की चेतावनी जारी की गई है। तुरंत निकलें।",
            "expected_hazard": True,
            "expected_type": "tsunami",
            "expected_location": "Chennai",
            "urgency": "high"
        },
        {
            "text": "मुंबई के समुद्री तट पर ऊंची लहरें आ रही हैं। सावधान रहें।",
            "expected_hazard": True,
            "expected_type": "high_waves",
            "expected_location": "Mumbai",
            "urgency": "medium"
        },
        {
            "text": "गोवा में आज मौसम बहुत अच्छा है। समुद्री किनारे पर घूमने के लिए बेहतरीन दिन।",
            "expected_hazard": False,
            "expected_type": "none",
            "expected_location": "Goa",
            "urgency": "low"
        }
    ],
    
    "tamil": [
        {
            "text": "சென்னை கடற்கரையில் சுனாமி எச்சரிக்கை. உடனே வெளியேறுங்கள்.",
            "expected_hazard": True,
            "expected_type": "tsunami",
            "expected_location": "Chennai",
            "urgency": "high"
        },
        {
            "text": "ராமேஸ்வரம் கடற்கரையில் பெரிய அலைகள். மீனவர்கள் கடலுக்குச் செல்ல வேண்டாம்.",
            "expected_hazard": True,
            "expected_type": "high_waves",
            "expected_location": "Rameswaram",
            "urgency": "medium"
        }
    ],
    
    "code_mixed": [
        {
            "text": "Mumbai beach pe bahut dangerous waves aa rahe hain! Please stay away 🌊",
            "expected_hazard": True,
            "expected_type": "high_waves",
            "expected_location": "Mumbai",
            "urgency": "high"
        },
        {
            "text": "Chennai marina mein tsunami ka warning hai yaar, very serious situation 😰",
            "expected_hazard": True,
            "expected_type": "tsunami",
            "expected_location": "Chennai",
            "urgency": "high"
        },
        {
            "text": "Goa beach mein aaj perfect weather hai, photo shoot ke liye awesome day 📸",
            "expected_hazard": False,
            "expected_type": "none",
            "expected_location": "Goa",
            "urgency": "low"
        }
    ]
}

# Edge Cases for Testing
EDGE_CASES = [
    {
        "category": "empty_input",
        "text": "",
        "expected_result": "validation_error"
    },
    {
        "category": "only_emojis",
        "text": "🌊🌊🌊😱😱😱",
        "expected_result": "no_content_extracted"
    },
    {
        "category": "very_long_text",
        "text": "Tsunami warning " * 100 + "for Chennai coast.",
        "expected_result": "processed_with_truncation"
    },
    {
        "category": "special_characters",
        "text": "Tsunami @#$%^&*() warning !!! Chennai $$$ coast ???",
        "expected_hazard": True,
        "expected_location": "Chennai"
    },
    {
        "category": "mixed_languages_complex",
        "text": "चेन्नई में tsunami का warning है and मुंबई also affected with high waves",
        "expected_hazard": True,
        "expected_locations": ["Chennai", "Mumbai"]
    },
    {
        "category": "ambiguous_location",
        "text": "Tsunami warning near the big temple by the sea",
        "expected_result": "location_ambiguous"
    },
    {
        "category": "misinformation",
        "text": "Scientists confirm aliens caused the tsunami in Chennai! Share this message to warn everyone!",
        "expected_hazard": True,
        "expected_misinformation": True,
        "expected_location": "Chennai"
    },
    {
        "category": "sarcasm",
        "text": "Oh great, another tsunami warning. Just what we needed today. Thanks Chennai weather!",
        "expected_hazard": True,
        "expected_sentiment": "sarcastic",
        "expected_location": "Chennai"
    }
]

# Performance Test Data
PERFORMANCE_TEST_DATA = {
    "single_requests": [
        {
            "text": f"Tsunami warning for location_{i}. Evacuate immediately.",
            "source": "automated_test",
            "timestamp": "2024-01-01T10:00:00Z"
        }
        for i in range(100)
    ],
    
    "batch_requests": [
        {
            "reports": [
                {
                    "text": f"High waves at beach_{j}_{i}",
                    "source": "test_batch",
                    "timestamp": f"2024-01-01T{10+i:02d}:00:00Z"
                }
                for j in range(10)
            ]
        }
        for i in range(20)
    ]
}

# Location Fuzzy Matching Test Cases
LOCATION_FUZZY_TESTS = [
    {
        "input": "Puri bech",
        "expected": "Puri Beach",
        "confidence_threshold": 0.8
    },
    {
        "input": "Chennai marina",
        "expected": "Marina Beach",
        "confidence_threshold": 0.9
    },
    {
        "input": "Mumbay juhu",
        "expected": "Juhu Beach",
        "confidence_threshold": 0.85
    },
    {
        "input": "Kochi fort cochin",
        "expected": "Fort Kochi",
        "confidence_threshold": 0.8
    },
    {
        "input": "Vizag beach",
        "expected": "Visakhapatnam",
        "confidence_threshold": 0.75
    }
]

# Anomaly Detection Test Scenarios
ANOMALY_TEST_SCENARIOS = [
    {
        "scenario": "spatial_cluster",
        "description": "Multiple reports from same location",
        "data": [
            {"lat": 13.08, "lon": 80.27, "time": "10:00", "type": "tsunami"},
            {"lat": 13.09, "lon": 80.28, "time": "10:05", "type": "tsunami"},
            {"lat": 13.07, "lon": 80.26, "time": "10:10", "type": "high_waves"},
            {"lat": 13.08, "lon": 80.29, "time": "10:15", "type": "tsunami"}
        ],
        "expected_anomaly": True,
        "anomaly_type": "spatial_cluster"
    },
    {
        "scenario": "temporal_spike",
        "description": "Sudden increase in reports",
        "data": [
            {"time": f"{8+i:02d}:00", "count": 1 if i < 10 else 15}
            for i in range(20)
        ],
        "expected_anomaly": True,
        "anomaly_type": "temporal_spike"
    },
    {
        "scenario": "normal_pattern",
        "description": "Regular reporting pattern",
        "data": [
            {"lat": 13.08 + i*0.1, "lon": 80.27 + i*0.1, "time": f"{10+i:02d}:00", "type": "normal"}
            for i in range(10)
        ],
        "expected_anomaly": False
    }
]

# API Load Testing Scenarios
LOAD_TEST_SCENARIOS = [
    {
        "name": "light_load",
        "concurrent_users": 5,
        "requests_per_user": 10,
        "ramp_up_time": 5,
        "expected_success_rate": 100
    },
    {
        "name": "moderate_load", 
        "concurrent_users": 20,
        "requests_per_user": 25,
        "ramp_up_time": 10,
        "expected_success_rate": 95
    },
    {
        "name": "heavy_load",
        "concurrent_users": 50,
        "requests_per_user": 50,
        "ramp_up_time": 30,
        "expected_success_rate": 90
    }
]

# Government Alert Test Cases
GOVERNMENT_ALERT_SAMPLES = [
    {
        "source": "INCOIS",
        "text": "Tsunami Advisory: A tsunami with wave heights of 0.3 to 1.0 meters is possible for the coasts of Tamil Nadu and Andhra Pradesh",
        "expected_hazard": True,
        "expected_type": "tsunami",
        "expected_severity": "medium",
        "expected_locations": ["Tamil Nadu", "Andhra Pradesh"]
    },
    {
        "source": "IMD",
        "text": "Very rough sea conditions along Karnataka coast. Fishermen advised not to venture into sea",
        "expected_hazard": True,
        "expected_type": "rough_sea",
        "expected_locations": ["Karnataka"]
    },
    {
        "source": "NDMA",
        "text": "Cyclone YAAS: Very heavy rainfall expected in West Bengal and Odisha. Storm surge height 2-4 meters",
        "expected_hazard": True,
        "expected_type": "cyclone",
        "expected_locations": ["West Bengal", "Odisha"]
    }
]

# Social Media Test Cases with Noise
SOCIAL_MEDIA_SAMPLES = [
    {
        "platform": "twitter",
        "text": "OMG! Huge waves at Juhu beach Mumbai 😱 Everyone running! #tsunami #mumbai #waves",
        "expected_hazard": True,
        "noise_level": "high",
        "urgency": "high"
    },
    {
        "platform": "facebook",
        "text": "Guys, be careful at Chennai beach today. Water level seems higher than usual. Might be dangerous.",
        "expected_hazard": True,
        "noise_level": "medium",
        "urgency": "medium"
    },
    {
        "platform": "instagram",
        "text": "Amazing waves for surfing at Goa beach today! Perfect conditions 🏄‍♂️ #surfing #goa #waves",
        "expected_hazard": False,
        "noise_level": "low",
        "context": "recreational"
    }
]

# Error Handling Test Cases
ERROR_HANDLING_TESTS = [
    {
        "test_type": "malformed_json",
        "data": '{"text": "test", "source": "test"',  # Missing closing brace
        "expected_error": "json_parse_error"
    },
    {
        "test_type": "missing_required_field",
        "data": {"source": "test"},  # Missing text field
        "expected_error": "validation_error"
    },
    {
        "test_type": "invalid_timestamp",
        "data": {
            "text": "test",
            "source": "test", 
            "timestamp": "invalid-date"
        },
        "expected_error": "timestamp_parse_error"
    },
    {
        "test_type": "extremely_large_text",
        "data": {
            "text": "A" * 100000,  # 100KB of text
            "source": "test"
        },
        "expected_behavior": "truncate_or_reject"
    }
]

# Multi-step Integration Test Scenarios
INTEGRATION_TEST_SCENARIOS = [
    {
        "name": "complete_hazard_flow",
        "steps": [
            {
                "action": "submit_report",
                "data": {
                    "text": "Tsunami warning for Chennai marina beach",
                    "source": "twitter"
                }
            },
            {
                "action": "verify_processing",
                "expected": {
                    "is_hazard": True,
                    "hazard_type": "tsunami",
                    "location_found": True
                }
            },
            {
                "action": "check_anomaly_detection",
                "expected": {
                    "anomaly_checked": True
                }
            },
            {
                "action": "export_data",
                "format": "geojson",
                "expected": {
                    "contains_report": True,
                    "valid_coordinates": True
                }
            }
        ]
    },
    {
        "name": "feedback_correction_flow",
        "steps": [
            {
                "action": "submit_false_positive",
                "data": {
                    "text": "High tide at Chennai beach, normal phenomenon",
                    "source": "user_report"
                }
            },
            {
                "action": "verify_wrong_classification",
                "expected": {
                    "is_hazard": True  # Wrong classification
                }
            },
            {
                "action": "submit_feedback",
                "correction": {
                    "is_hazard": False,
                    "hazard_type": "none"
                }
            },
            {
                "action": "verify_feedback_recorded",
                "expected": {
                    "feedback_accepted": True
                }
            }
        ]
    }
]

if __name__ == "__main__":
    print("Test data samples loaded successfully!")
    print(f"Available test categories:")
    print(f"- Multilingual samples: {len(MULTILINGUAL_SAMPLES)} languages")
    print(f"- Edge cases: {len(EDGE_CASES)} scenarios")
    print(f"- Performance data: {len(PERFORMANCE_TEST_DATA['single_requests'])} single requests")
    print(f"- Location fuzzy tests: {len(LOCATION_FUZZY_TESTS)} cases")
    print(f"- Anomaly scenarios: {len(ANOMALY_TEST_SCENARIOS)} scenarios")
    print(f"- Load test scenarios: {len(LOAD_TEST_SCENARIOS)} scenarios")
    print(f"- Government alerts: {len(GOVERNMENT_ALERT_SAMPLES)} samples")
    print(f"- Social media: {len(SOCIAL_MEDIA_SAMPLES)} samples")
    print(f"- Error handling: {len(ERROR_HANDLING_TESTS)} cases")
    print(f"- Integration scenarios: {len(INTEGRATION_TEST_SCENARIOS)} scenarios")