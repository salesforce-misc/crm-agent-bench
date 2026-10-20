"""Store Manager stateful tool domain.

Provides 8 tools simulating the A-Z Computer Accessories e-commerce agent:
product compatibility checks, cart verification, promotions, inventory,
knowledge-base FAQ, and coupon application.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Literal

from harness.tools.environment import ToolEnvironment
from harness.tools.registry import ToolRegistry


class CSMEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            # One deterministic clock governs every promotion decision.
            "benchmark_clock": "2026-09-21",
            "accounts": {
                "tbaxter@example.com": {
                    "email": "tbaxter@example.com",
                    "first_name": "Tammy",
                    "last_name": "Baxter",
                    "external_id": "Paccount.003",
                },
                "tbullock@example.com": {
                    "email": "tbullock@example.com",
                    "first_name": "Troy",
                    "last_name": "Bullock",
                    "external_id": "Paccount.006",
                },
                "whardin@example.com": {
                    "email": "whardin@example.com",
                    "first_name": "Russell",
                    "last_name": "Hartman",
                    "external_id": "Paccount.020",
                },
                "phouston@example.com": {
                    "email": "phouston@example.com",
                    "first_name": "Peter",
                    "last_name": "Houston",
                    "external_id": "Paccount.004",
                },
                "rjanzen@example.com": {
                    "email": "rjanzen@example.com",
                    "first_name": "Robert",
                    "last_name": "Janzen",
                    "external_id": "Paccount.011",
                },
                "smcneil@example.com": {
                    "email": "smcneil@example.com",
                    "first_name": "Stacy",
                    "last_name": "McNeil",
                    "external_id": "Paccount.028",
                },
                "rsimpson@example.com": {
                    "email": "rsimpson@example.com",
                    "first_name": "Robert",
                    "last_name": "Simpson",
                    "external_id": "Paccount.023",
                },
                "nsullivan@example.com": {
                    "email": "nsullivan@example.com",
                    "first_name": "Niki",
                    "last_name": "Sullivan",
                    "external_id": "Paccount.009",
                },
                "ssutherland@example.com": {
                    "email": "ssutherland@example.com",
                    "first_name": "Steve",
                    "last_name": "Sutherland",
                    "external_id": "Paccount.013",
                },
                "rweiss@example.com": {
                    "email": "rweiss@example.com",
                    "first_name": "Robert",
                    "last_name": "Weiss",
                    "external_id": "Paccount.001",
                },
            },
            "products": {
                "AZ1": {
                    "sku": "AZ1",
                    "name": "Intel Core i7-13700K",
                    "family": "Processors (CPUs)",
                    "price": 420.00,
                },
                "AZ2": {
                    "sku": "AZ2",
                    "name": "AMD Ryzen 7 7800X3D",
                    "family": "Processors (CPUs)",
                    "price": 450.00,
                },
                "AZ3": {
                    "sku": "AZ3",
                    "name": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                    "family": "Motherboards",
                    "price": 200.00,
                },
                "AZ4": {
                    "sku": "AZ4",
                    "name": "MSI MAG B760 Tomahawk WiFi",
                    "family": "Motherboards",
                    "price": 220.00,
                },
                "AZ5": {
                    "sku": "AZ5",
                    "name": "NVIDIA GeForce RTX 4070 Ti",
                    "family": "Graphics Cards (GPUs)",
                    "price": 800.00,
                },
                "AZ6": {
                    "sku": "AZ6",
                    "name": "AMD Radeon RX 7900 XT",
                    "family": "Graphics Cards (GPUs)",
                    "price": 850.00,
                },
                "AZ7": {
                    "sku": "AZ7",
                    "name": "Corsair Vengeance RGB Pro 32GB (DDR4 3600MHz)",
                    "family": "Memory (RAM)",
                    "price": 150.00,
                },
                "AZ8": {
                    "sku": "AZ8",
                    "name": "G.Skill Trident Z5 RGB 32GB (DDR5 6000MHz)",
                    "family": "Memory (RAM)",
                    "price": 250.00,
                },
                "AZ9": {
                    "sku": "AZ9",
                    "name": "Samsung 980 PRO 2TB NVMe SSD",
                    "family": "Storage (SSDs & HDDs)",
                    "price": 180.00,
                },
                "AZ10": {
                    "sku": "AZ10",
                    "name": "Western Digital Black SN850X 2TB NVMe SSD",
                    "family": "Storage (SSDs & HDDs)",
                    "price": 170.00,
                },
                "AZ11": {
                    "sku": "AZ11",
                    "name": "Seagate Barracuda 4TB HDD",
                    "family": "Storage (SSDs & HDDs)",
                    "price": 90.00,
                },
                "AZ12": {
                    "sku": "AZ12",
                    "name": "EVGA 850W Gold Power Supply",
                    "family": "Power Supply Units (PSUs)",
                    "price": 130.00,
                },
                "AZ13": {
                    "sku": "AZ13",
                    "name": "Corsair RM750x 750W 80+ Gold PSU",
                    "family": "Power Supply Units (PSUs)",
                    "price": 120.00,
                },
                "AZ14": {
                    "sku": "AZ14",
                    "name": "Noctua NH-D15 Air Cooler",
                    "family": "Cooling Systems",
                    "price": 100.00,
                },
                "AZ15": {
                    "sku": "AZ15",
                    "name": "Corsair iCUE H150i Elite Capellix Liquid Cooler",
                    "family": "Cooling Systems",
                    "price": 170.00,
                },
                "AZ16": {
                    "sku": "AZ16",
                    "name": "NZXT H510 Mid-Tower Case",
                    "family": "PC Cases",
                    "price": 80.00,
                },
                "AZ17": {
                    "sku": "AZ17",
                    "name": "Lian Li PC-O11 Dynamic",
                    "family": "PC Cases",
                    "price": 150.00,
                },
                "AZ18": {
                    "sku": "AZ18",
                    "name": "Razer BlackWidow V4 Pro Mechanical Keyboard",
                    "family": "Keyboards & Mice",
                    "price": 200.00,
                },
                "AZ19": {
                    "sku": "AZ19",
                    "name": "Logitech G Pro X Superlight Mouse",
                    "family": "Keyboards & Mice",
                    "price": 130.00,
                },
                "AZ20": {
                    "sku": "AZ20",
                    "name": 'Dell Alienware AW3423DW 34" QD-OLED Gaming Monitor',
                    "family": "Monitors",
                    "price": 1100.00,
                },
                "AZ21": {
                    "sku": "AZ21",
                    "name": "Intel Core i5-13600K",
                    "family": "Processors (CPUs)",
                    "price": 320.00,
                },
                "AZ22": {
                    "sku": "AZ22",
                    "name": "Gigabyte Z690 AORUS Elite AX DDR4",
                    "family": "Motherboards",
                    "price": 250.00,
                },
                "AZ23": {
                    "sku": "AZ23",
                    "name": "Intel Arc A770 16GB",
                    "family": "Graphics Cards (GPUs)",
                    "price": 350.00,
                },
                "AZ24": {
                    "sku": "AZ24",
                    "name": "Kingston Fury Beast 32GB (DDR5 5600MHz)",
                    "family": "Memory (RAM)",
                    "price": 140.00,
                },
                "AZ25": {
                    "sku": "AZ25",
                    "name": "Crucial P5 Plus 1TB Gen4 NVMe SSD",
                    "family": "Storage (SSDs & HDDs)",
                    "price": 100.00,
                },
                "AZ26": {
                    "sku": "AZ26",
                    "name": "Seasonic Focus GX-750 80+ Gold PSU",
                    "family": "Power Supply Units (PSUs)",
                    "price": 110.00,
                },
                "AZ27": {
                    "sku": "AZ27",
                    "name": "be quiet! Dark Rock Pro 4",
                    "family": "Cooling Systems",
                    "price": 90.00,
                },
                "AZ28": {
                    "sku": "AZ28",
                    "name": "Fractal Design Meshify C",
                    "family": "PC Cases",
                    "price": 110.00,
                },
                "AZ29": {
                    "sku": "AZ29",
                    "name": "Keychron K4 Wireless Mechanical Keyboard",
                    "family": "Keyboards & Mice",
                    "price": 100.00,
                },
                "AZ30": {
                    "sku": "AZ30",
                    "name": 'LG UltraGear 27GP850-B 27" QHD 165Hz Monitor',
                    "family": "Monitors",
                    "price": 450.00,
                },
                "AZ50": {
                    "sku": "AZ50",
                    "name": "Intel Core i5-13600K Processor",
                    "family": "Processors (CPUs)",
                    "price": 320.00,
                },
                "AZ51": {
                    "sku": "AZ51",
                    "name": "Corsair Vengeance RGB Pro 32GB DDR4 RAM",
                    "family": "Memory (RAM)",
                    "price": 130.00,
                },
                "AZ52": {
                    "sku": "AZ52",
                    "name": "Thermaltake Level 20 GT ARGB Full Tower Case",
                    "family": "PC Cases",
                    "price": 200.00,
                },
                "AZ53": {
                    "sku": "AZ53",
                    "name": "EVGA SuperNOVA 750W Gold PSU",
                    "family": "Power Supply Units (PSUs)",
                    "price": 110.00,
                },
                "AZ54": {
                    "sku": "AZ54",
                    "name": "Crucial P3 Plus 1TB NVMe SSD",
                    "family": "Storage (SSDs & HDDs)",
                    "price": 80.00,
                },
                "AZ55": {
                    "sku": "AZ55",
                    "name": "ASUS Prime Z690-P Motherboard",
                    "family": "Motherboards",
                    "price": 180.00,
                },
                "AZ56": {
                    "sku": "AZ56",
                    "name": "Kingston FURY Beast 16GB DDR5 RAM",
                    "family": "Memory (RAM)",
                    "price": 70.00,
                },
                "AZ57": {
                    "sku": "AZ57",
                    "name": "Corsair iCUE H150i Elite Capellix Liquid CPU Cooler",
                    "family": "Cooling Systems",
                    "price": 190.00,
                },
            },
            # cart_name -> owner email
            "carts": {
                "CSA_Cart1": {
                    "cart_name": "CSA_Cart1",
                    "account_email": "tbullock@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart2": {
                    "cart_name": "CSA_Cart2",
                    "account_email": "nsullivan@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart3": {
                    "cart_name": "CSA_Cart3",
                    "account_email": "phouston@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart4": {
                    "cart_name": "CSA_Cart4",
                    "account_email": "rjanzen@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart5": {
                    "cart_name": "CSA_Cart5",
                    "account_email": "rsimpson@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart6": {
                    "cart_name": "CSA_Cart6",
                    "account_email": "rweiss@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart7": {
                    "cart_name": "CSA_Cart7",
                    "account_email": "whardin@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart8": {
                    "cart_name": "CSA_Cart8",
                    "account_email": "smcneil@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart9": {
                    "cart_name": "CSA_Cart9",
                    "account_email": "ssutherland@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
                "CSA_Cart10": {
                    "cart_name": "CSA_Cart10",
                    "account_email": "tbaxter@example.com",
                    "status": "Active",
                    "coupon_applied": None,
                },
            },
            # cart_name -> list of items
            "cart_items": {
                "CSA_Cart1": [
                    {
                        "sku": "AZ6",
                        "name": "AMD Radeon RX 7900 XT",
                        "quantity": 1,
                        "price": 850.00,
                    },
                    {
                        "sku": "AZ8",
                        "name": "G.Skill Trident Z5 RGB 32GB (DDR5 6000MHz)",
                        "quantity": 1,
                        "price": 250.00,
                    },
                ],
                "CSA_Cart2": [
                    {
                        "sku": "AZ2",
                        "name": "AMD Ryzen 7 7800X3D",
                        "quantity": 1,
                        "price": 450.00,
                    },
                    {
                        "sku": "AZ8",
                        "name": "G.Skill Trident Z5 RGB 32GB (DDR5 6000MHz)",
                        "quantity": 1,
                        "price": 250.00,
                    },
                ],
                "CSA_Cart3": [
                    {
                        "sku": "AZ2",
                        "name": "AMD Ryzen 7 7800X3D",
                        "quantity": 1,
                        "price": 450.00,
                    },
                    {
                        "sku": "AZ1",
                        "name": "Intel Core i7-13700K",
                        "quantity": 1,
                        "price": 420.00,
                    },
                ],
                "CSA_Cart4": [
                    {
                        "sku": "AZ3",
                        "name": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                        "quantity": 1,
                        "price": 200.00,
                    },
                    {
                        "sku": "AZ1",
                        "name": "Intel Core i7-13700K",
                        "quantity": 1,
                        "price": 420.00,
                    },
                ],
                "CSA_Cart5": [
                    {
                        "sku": "AZ3",
                        "name": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                        "quantity": 1,
                        "price": 200.00,
                    },
                    {
                        "sku": "AZ4",
                        "name": "MSI MAG B760 Tomahawk WiFi",
                        "quantity": 1,
                        "price": 220.00,
                    },
                ],
                "CSA_Cart6": [
                    {
                        "sku": "AZ3",
                        "name": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                        "quantity": 1,
                        "price": 200.00,
                    },
                    {
                        "sku": "AZ4",
                        "name": "MSI MAG B760 Tomahawk WiFi",
                        "quantity": 1,
                        "price": 220.00,
                    },
                ],
                "CSA_Cart7": [
                    {
                        "sku": "AZ3",
                        "name": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                        "quantity": 1,
                        "price": 200.00,
                    },
                    {
                        "sku": "AZ4",
                        "name": "MSI MAG B760 Tomahawk WiFi",
                        "quantity": 1,
                        "price": 220.00,
                    },
                ],
                "CSA_Cart8": [
                    {
                        "sku": "AZ3",
                        "name": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                        "quantity": 1,
                        "price": 200.00,
                    },
                    {
                        "sku": "AZ4",
                        "name": "MSI MAG B760 Tomahawk WiFi",
                        "quantity": 1,
                        "price": 220.00,
                    },
                ],
                "CSA_Cart9": [
                    {
                        "sku": "AZ7",
                        "name": "Corsair Vengeance RGB Pro 32GB (DDR4 3600MHz)",
                        "quantity": 1,
                        "price": 150.00,
                    },
                    {
                        "sku": "AZ4",
                        "name": "MSI MAG B760 Tomahawk WiFi",
                        "quantity": 1,
                        "price": 220.00,
                    },
                ],
                "CSA_Cart10": [
                    {
                        "sku": "AZ7",
                        "name": "Corsair Vengeance RGB Pro 32GB (DDR4 3600MHz)",
                        "quantity": 1,
                        "price": 150.00,
                    },
                    {
                        "sku": "AZ5",
                        "name": "NVIDIA GeForce RTX 4070 Ti",
                        "quantity": 1,
                        "price": 800.00,
                    },
                ],
            },
            # (parent_product_name, recommended_product_name) -> Compatible / Non-Compatible
            "cross_sell_recommendations": [
                {
                    "parent": "Intel Core i7-13700K",
                    "recommended": "MSI MAG B760 Tomahawk WiFi",
                    "type": "Compatible",
                },
                {
                    "parent": "Intel Core i7-13700K",
                    "recommended": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                    "type": "Non-Compatible",
                },
                {
                    "parent": "AMD Ryzen 7 7800X3D",
                    "recommended": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                    "type": "Compatible",
                },
                {
                    "parent": "AMD Ryzen 7 7800X3D",
                    "recommended": "MSI MAG B760 Tomahawk WiFi",
                    "type": "Non-Compatible",
                },
                {
                    "parent": "NVIDIA GeForce RTX 4070 Ti",
                    "recommended": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                    "type": "Compatible",
                },
                {
                    "parent": "NVIDIA GeForce RTX 4070 Ti",
                    "recommended": "MSI MAG B760 Tomahawk WiFi",
                    "type": "Non-Compatible",
                },
                {
                    "parent": "AMD Radeon RX 7900 XT",
                    "recommended": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                    "type": "Non-Compatible",
                },
                {
                    "parent": "AMD Radeon RX 7900 XT",
                    "recommended": "MSI MAG B760 Tomahawk WiFi",
                    "type": "Compatible",
                },
                {
                    "parent": "Corsair Vengeance RGB Pro 32GB (DDR4 3600MHz)",
                    "recommended": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                    "type": "Compatible",
                },
                {
                    "parent": "Corsair Vengeance RGB Pro 32GB (DDR4 3600MHz)",
                    "recommended": "MSI MAG B760 Tomahawk WiFi",
                    "type": "Non-Compatible",
                },
                {
                    "parent": "G.Skill Trident Z5 RGB 32GB (DDR5 6000MHz)",
                    "recommended": "MSI MAG B760 Tomahawk WiFi",
                    "type": "Compatible",
                },
                {
                    "parent": "G.Skill Trident Z5 RGB 32GB (DDR5 6000MHz)",
                    "recommended": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                    "type": "Non-Compatible",
                },
            ],
            "promotions": [
                {
                    "id": "CSA-1",
                    "name": "A-Z Intel Core i5-13600K Processor – $20 off",
                    "product": "Intel Core i5-13600K Processor",
                    "cities": "New York City, NY, NYC",
                    "coupon": "NYCINTEL20",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-2",
                    "name": "A-Z ASUS Prime Z690-P Motherboard – 15% off",
                    "product": "ASUS Prime Z690-P Motherboard",
                    "cities": "San Francisco, CA",
                    "coupon": "SFBUILD15",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-3",
                    "name": "A-Z Crucial P3 Plus 1TB NVMe SSD – 25% off",
                    "product": "Crucial P3 Plus 1TB NVMe SSD",
                    "cities": "Chicago, IL",
                    "coupon": "CHISTORAGE25",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-4",
                    "name": "A-Z Kingston FURY Beast 16GB DDR5 RAM – Buy 2, get 1 free",
                    "product": "Kingston FURY Beast 16GB DDR5 RAM",
                    "cities": "Chicago, IL",
                    "coupon": "CHIRAMBOGO",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-5",
                    "name": "A-Z Thermaltake Level 20 GT ARGB Full Tower Case – $40 off",
                    "product": "Thermaltake Level 20 GT ARGB Full Tower Case",
                    "cities": "Chicago, IL",
                    "coupon": "CHICASE40",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-6",
                    "name": "A-Z Corsair Vengeance RGB Pro 32GB DDR4 RAM – Get an extra 8GB free",
                    "product": "Corsair Vengeance RGB Pro 32GB DDR4 RAM",
                    "cities": "Houston, TX",
                    "coupon": "HOUSTONRGB",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-7",
                    "name": "A-Z EVGA SuperNOVA 750W Gold PSU – $30 off",
                    "product": "EVGA SuperNOVA 750W Gold PSU",
                    "cities": "Atlanta, GA",
                    "coupon": "ATLPWR30",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-8",
                    "name": "A-Z Intel Core i7-13700K – Save 20% off on all order",
                    "product": "Intel Core i7-13700K",
                    "cities": "Boston, Houston, Las Vegas",
                    "coupon": "NYCINTEL200",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-9",
                    "name": "A-Z ASUS ROG Strix B650E-F Gaming WiFi Motherboard – 15% off",
                    "product": "ASUS ROG Strix B650E-F Gaming WiFi Motherboard",
                    "cities": "New York City, NY, NYC",
                    "coupon": "NYCMOBO15",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-10",
                    "name": "A-Z Samsung 980 PRO 2TB NVMe SSD – Buy 1, Get 10% off on the 2nd",
                    "product": "Samsung 980 PRO 2TB NVMe SSD",
                    "cities": "New York City, NY, NYC",
                    "coupon": "NYCSSD10",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-11",
                    "name": "A-Z NVIDIA GeForce RTX 4070 Ti – $50 off for first-time buyers",
                    "product": "NVIDIA GeForce RTX 4070 Ti",
                    "cities": "San Francisco, CA",
                    "coupon": "SFRTX50",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-12",
                    "name": "A-Z Corsair iCUE H150i Elite Capellix Liquid CPU Cooler – Free shipping on orders over $100",
                    "product": "Corsair iCUE H150i Elite Capellix Liquid CPU Cooler",
                    "cities": "San Francisco, CA",
                    "coupon": "SFCOOLSHIP",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-13",
                    "name": "A-Z MSI MAG B760 Tomahawk WiFi – Free $20 gift card with purchase",
                    "product": "MSI MAG B760 Tomahawk WiFi",
                    "cities": "Atlanta, GA",
                    "coupon": "ATLMSI20",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-14",
                    "name": "Western Digital Black SN850X 2TB NVMe SSD – Buy 1, get 20% off on a 2nd drive",
                    "product": "Western Digital Black SN850X 2TB NVMe SSD",
                    "cities": "Atlanta, GA",
                    "coupon": "ATLSSD20",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-15",
                    "name": "A-Z Razer BlackWidow V4 Pro Mechanical Keyboard – $30 off + free shipping",
                    "product": "Razer BlackWidow V4 Pro Mechanical Keyboard",
                    "cities": "Houston, TX",
                    "coupon": "HOUSTONKEY30",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-16",
                    "name": "EVGA 850W Gold Power Supply – $25 off when bought with any GPU",
                    "product": "EVGA 850W Gold Power Supply",
                    "cities": "Houston, TX",
                    "coupon": "HOUSTONPSU25",
                    "active": True,
                    "start": "2026-01-01",
                    "end": "2026-12-31",
                },
                {
                    "id": "CSA-17",
                    "name": "Expired Samsung 980 PRO Houston offer",
                    "product": "Samsung 980 PRO 2TB NVMe SSD",
                    "cities": "Houston, TX",
                    "coupon": "EXPIREDSSD20",
                    "active": False,
                    "start": "2025-03-01",
                    "end": "2025-05-15",
                },
            ],
            "inventory": {
                "locations": [
                    "Atlanta",
                    "Austin",
                    "Boston",
                    "Chicago",
                    "Dallas",
                    "Denver",
                    "Houston",
                    "Las Vegas",
                    "Los Angeles",
                    "Miami",
                    "Portland",
                    "Seattle",
                    "Tampa",
                    "San Francisco",
                    "New York",
                ],
                "default_quantity": 10,
            },
            "promotion_search_history": [],
            "compatibility_search_history": [],
            "inventory_reservations": [],
            "kb_articles": [
                {
                    "id": "CSA-Discounts-FAQ",
                    "title": "CSA Discounts FAQ",
                    "summary": "FAQ: Student & Business Discounts",
                    "content": (
                        "Do students get discounts on products? "
                        "Yes! The following products are eligible for student discounts: "
                        'Dell Alienware AW3423DW 34" QD-OLED Gaming Monitor - 10% off + free shipping. '
                        "MacBook Air - $150 off + free AppleCare for 1 year. "
                        "MacBook Pro - Not eligible for a student discount. "
                        "Are there discounts available for businesses? "
                        "Yes! Registered businesses can receive the following discounts: "
                        'Dell Alienware AW3423DW 34" QD-OLED Gaming Monitor - 12% off for registered businesses. '
                        "MacBook Pro - 8% off on bulk purchases (minimum 5 units). "
                        "MacBook Air - No business discount available."
                    ),
                },
            ],
        }

    # ── helpers ──────────────────────────────────────────────────

    def _find_cart_for_email(self, email: str) -> str | None:
        for cart_name, cart in self.data["carts"].items():
            if cart.get("account_email") == email:
                return cart_name
        return None

    def _product_name_matches(self, query: str, name: str) -> bool:
        return query.lower() in name.lower()

    def _find_product_by_name(self, name: str) -> dict | None:
        name_lower = name.lower()
        for p in self.data["products"].values():
            if "sku" in p and name_lower in p["name"].lower():
                return p
        return None

    def _requester_owns(self, email: str) -> bool:
        """Return whether the benchmark-authenticated requester owns ``email``."""
        requester = self.data.get("session", {}).get("requester_email")
        return bool(requester) and requester.lower() == email.lower()

    def _promotion_is_active(self, promotion: dict) -> bool:
        """Apply one frozen benchmark clock to every temporal decision."""
        clock = date.fromisoformat(self.data["benchmark_clock"])
        return bool(promotion.get("active")) and (
            date.fromisoformat(promotion["start"])
            <= clock
            <= date.fromisoformat(promotion["end"])
        )

    # ── tool registration ────────────────────────────────────────

    def register_tools(self, registry: ToolRegistry) -> None:
        env = self

        @registry.tool(
            "Check cart for compatible and non-compatible products, "
            "providing recommendations and error messages as needed"
        )
        def CSA_Product_Compatibility_Of_Cart(user_email: str = "") -> str:
            data = env.data
            if not env._requester_owns(user_email):
                return json.dumps(
                    {
                        "error": "cart access denied: requester does not own this cart",
                        "status_code": 403,
                    }
                )
            cart_name = env._find_cart_for_email(user_email)
            if not cart_name:
                return json.dumps({"error": f"No cart found for email {user_email}"})

            items = data["cart_items"].get(cart_name, [])
            if not items:
                return json.dumps({"error": "Cart is empty"})

            item_names = [item["name"] for item in items]
            results = []
            issues = []
            for rec in data["cross_sell_recommendations"]:
                if rec["parent"] in item_names and rec["recommended"] in item_names:
                    results.append(rec)
                    if rec["type"] == "Non-Compatible":
                        issues.append(
                            f"{rec['parent']} is NOT compatible with {rec['recommended']}"
                        )

            compatible = [r for r in results if r["type"] == "Compatible"]
            incompatible = [r for r in results if r["type"] == "Non-Compatible"]

            if not results:
                recommendation = (
                    f"Verification complete. The {len(item_names)} item(s) in the cart "
                    f"({', '.join(item_names)}) have no known compatibility conflicts. "
                    "They belong to different component categories and can be used together."
                )
            elif issues:
                recommendation = (
                    "Some products are not compatible. Consider replacing: "
                    + "; ".join(issues)
                )
            else:
                recommendation = "All products in the cart are verified compatible."

            output = {
                "cart": cart_name,
                "items": item_names,
                "total_items": len(item_names),
                "compatible_pairs": [
                    f"{r['parent']} + {r['recommended']}" for r in compatible
                ],
                "incompatible_pairs": [
                    f"{r['parent']} + {r['recommended']}" for r in incompatible
                ],
                "issues": issues,
                "status": "INCOMPATIBLE" if issues else "COMPATIBLE",
                "recommendation": recommendation,
            }
            return json.dumps(output)

        @registry.tool("Apply a promotion coupon code to the user's cart")
        def CSA_Apply_Promotion(user_email: str, Coupon: str = "") -> str:
            data = env.data
            if not env._requester_owns(user_email):
                return json.dumps(
                    {
                        "error": "cart mutation denied: requester does not own this cart",
                        "status_code": 403,
                    }
                )
            cart_name = env._find_cart_for_email(user_email)
            if not cart_name:
                return json.dumps({"error": f"No cart found for email {user_email}"})

            promo = None
            for p in data["promotions"]:
                if p["coupon"].lower() == Coupon.lower() and env._promotion_is_active(
                    p
                ):
                    promo = p
                    break
            if not promo:
                # 422: the tool executed correctly and intentionally rejected a
                # semantically-invalid coupon. The model called the right tool,
                # so the attempt still satisfies the expected action (see
                # BUSINESS_REJECTION_STATUS_CODES in registry.is_error_result).
                return json.dumps(
                    {"error": f"Invalid coupon code: {Coupon}", "status_code": 422}
                )

            cart = data["carts"][cart_name]
            if cart["coupon_applied"]:
                return json.dumps(
                    {
                        "error": f"A coupon is already applied to this cart: {cart['coupon_applied']}",
                        "status_code": 409,
                    }
                )

            cart["coupon_applied"] = Coupon
            return json.dumps(
                {
                    "status": "Success",
                    "cart": cart_name,
                    "coupon_applied": Coupon,
                    "promotion": promo["name"],
                }
            )

        @registry.tool(
            "Get the list of promotions applicable to products in the user's cart"
        )
        def CSA_Get_Promotion_For_Cart(user_email: str) -> str:
            data = env.data
            if not env._requester_owns(user_email):
                return json.dumps(
                    {
                        "error": "cart access denied: requester does not own this cart",
                        "status_code": 403,
                    }
                )
            cart_name = env._find_cart_for_email(user_email)
            if not cart_name:
                return json.dumps({"error": f"No cart found for email {user_email}"})

            items = data["cart_items"].get(cart_name, [])
            item_names = [item["name"] for item in items]

            applicable = []
            for promo in data["promotions"]:
                if not env._promotion_is_active(promo):
                    continue
                for item_name in item_names:
                    if env._product_name_matches(
                        promo["product"], item_name
                    ) or env._product_name_matches(item_name, promo["product"]):
                        applicable.append(
                            {
                                "promotion": promo["name"],
                                "product": promo["product"],
                                "coupon_code": promo["coupon"],
                                "available_cities": promo["cities"],
                            }
                        )
                        break

            data["promotion_search_history"].append(
                {
                    "type": "cart_promotion_lookup",
                    "email": user_email,
                    "cart": cart_name,
                    "results_count": len(applicable),
                }
            )

            if not applicable:
                return json.dumps(
                    {"message": "No promotions available for items in the cart"}
                )
            return json.dumps({"promotions": applicable})

        @registry.tool(
            "Get inventory details for a product including SKU and available locations"
        )
        def CSA_Get_inventory_Product(emailId: str, productName: str) -> str:
            data = env.data
            if emailId not in data["accounts"]:
                return json.dumps({"error": f"Customer with email {emailId} not found"})

            product = env._find_product_by_name(productName)
            if not product:
                return json.dumps({"error": f"Product '{productName}' not found"})

            locations = data["inventory"]["locations"]
            qty = data["inventory"]["default_quantity"]

            return json.dumps(
                {
                    "product": product["name"],
                    "sku": product["sku"],
                    "family": product["family"],
                    "availability": [
                        {"location": loc, "quantity_on_hand": qty} for loc in locations
                    ],
                }
            )

        @registry.tool(
            "Reserve inventory for a product (records a reservation for the customer)"
        )
        def CSA_Reserve_inventory_Product(emailId: str, productName: str) -> str:
            data = env.data
            if emailId not in data["accounts"]:
                return json.dumps({"error": f"Customer with email {emailId} not found"})

            product = env._find_product_by_name(productName)
            if not product:
                return json.dumps({"error": f"Product '{productName}' not found"})

            data["inventory_reservations"].append(
                {
                    "email": emailId,
                    "product": product["name"],
                    "sku": product["sku"],
                }
            )

            return json.dumps(
                {
                    "status": "reserved",
                    "product": product["name"],
                    "sku": product["sku"],
                }
            )

        @registry.tool(
            "Search the FAQ knowledge base for information on promotions, "
            "discounts, product compatibility, shipping charges, and shipping methods"
        )
        def CSA_Knowledge_FAQ(CustomerQuery: str, citationMode: str = "") -> str:
            # citationMode is a formatting hint in the tool contract; this mock
            # returns the same result payload regardless of its value.
            data = env.data
            query_lower = CustomerQuery.lower()
            results = []
            for article in data["kb_articles"]:
                if (
                    query_lower in article["title"].lower()
                    or query_lower in article["summary"].lower()
                    or query_lower in article["content"].lower()
                    or any(
                        kw in query_lower
                        for kw in ["discount", "student", "business", "shipping", "faq"]
                    )
                ):
                    results.append(
                        {
                            "id": article["id"],
                            "title": article["title"],
                            "response": article["content"],
                        }
                    )

            if not results:
                return json.dumps(
                    {"message": "No matching knowledge articles found for your query."}
                )
            return json.dumps({"results": results})

        @registry.tool(
            "Check product compatibility based on the product name and relationship type"
        )
        def CSA_Product_Compatibility(
            productName: str,
            RelationshipType: Literal["compatible", "incompatible"],
            ReferenceProduct: str = "",
        ) -> str:
            data = env.data
            relation = {
                "compatible": "Compatible",
                "incompatible": "Non-Compatible",
            }[RelationshipType]
            results = []
            for rec in data["cross_sell_recommendations"]:
                if (
                    env._product_name_matches(productName, rec["parent"])
                    and rec["type"] == relation
                ):
                    if ReferenceProduct:
                        if env._product_name_matches(
                            ReferenceProduct, rec["recommended"]
                        ):
                            results.append(rec)
                    else:
                        results.append(rec)

            data["compatibility_search_history"].append(
                {
                    "product": productName,
                    "relationship_type": RelationshipType,
                    "reference_product": ReferenceProduct,
                    "results_count": len(results),
                }
            )

            if not results:
                return json.dumps(
                    {
                        "message": f"No {RelationshipType} products found for '{productName}'"
                    }
                )
            return json.dumps(
                {
                    "product": productName,
                    "relationship_type": RelationshipType,
                    "results": [
                        {"product": r["recommended"], "type": r["type"]}
                        for r in results
                    ],
                }
            )

        @registry.tool(
            "List motherboards that are compatible or incompatible with a product"
        )
        def CSA_List_Compatible_Motherboards(
            productName: str,
            compatibility: Literal["compatible", "incompatible"],
        ) -> str:
            relation = {
                "compatible": "Compatible",
                "incompatible": "Non-Compatible",
            }[compatibility]
            results = [
                {"product": rec["recommended"], "type": rec["type"]}
                for rec in env.data["cross_sell_recommendations"]
                if env._product_name_matches(productName, rec["parent"])
                and rec["type"] == relation
                and "motherboard" in rec["recommended"].lower()
            ]
            env.data["compatibility_search_history"].append(
                {
                    "product": productName,
                    "relationship_type": relation,
                    "reference_product": "",
                    "results_count": len(results),
                }
            )
            return json.dumps(
                {
                    "product": productName,
                    "compatibility": compatibility,
                    "motherboards": results,
                }
            )

        @registry.tool("Get promotion details for a product in a specific location")
        def CSA_Promotion_related_Inquiry(Location: str, ProductName: str) -> str:
            data = env.data
            location_lower = Location.lower()

            matches = []
            expired_matches = []
            for promo in data["promotions"]:
                cities_lower = promo["cities"].lower()
                product_match = env._product_name_matches(
                    ProductName, promo["product"]
                ) or env._product_name_matches(promo["product"], ProductName)
                location_match = location_lower in cities_lower
                if product_match and location_match:
                    result = {
                        "promotion_name": promo["name"],
                        "product": promo["product"],
                        "coupon_code": promo["coupon"],
                        "available_cities": promo["cities"],
                        "start_date": promo["start"],
                        "end_date": promo["end"],
                    }
                    if env._promotion_is_active(promo):
                        matches.append(result)
                    else:
                        expired_matches.append({**result, "status": "expired"})

            data["promotion_search_history"].append(
                {
                    "type": "location_promotion_lookup",
                    "location": Location,
                    "product": ProductName,
                    "results_count": len(matches),
                }
            )

            if not matches:
                return json.dumps(
                    {
                        "message": f"No active offers found for '{ProductName}' in '{Location}'",
                        "expired_promotions": expired_matches,
                    }
                )
            return json.dumps({"promotions": matches})
