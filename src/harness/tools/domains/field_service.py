"""Field Service Agent stateful tool domain.

Provides tools simulating a laptop repair field service operation:
FAQ knowledge base, technician user identification, troubleshooting guides,
appointment scheduling, work order management, and pre-work briefs.
"""

from __future__ import annotations

import json
import random

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


def _appointment_number() -> str:
    return f"SA-{random.randint(1000, 9999):04d}"


@register_domain("field_service")
class FieldServiceEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "users": {
                "rjanzen@example.com": {
                    "email": "rjanzen@example.com",
                    "first_name": "Robert",
                    "last_name": "Janzen",
                    "type": "Customer",
                    "account_id": "Paccount.011",
                },
                "ghoskins@example.com": {
                    "email": "ghoskins@example.com",
                    "first_name": "Gabe",
                    "last_name": "Hoskins",
                    "type": "Customer",
                    "account_id": "Paccount.024",
                },
                "mmalmin@example.com": {
                    "email": "mmalmin@example.com",
                    "first_name": "Mike",
                    "last_name": "Malmin",
                    "type": "Customer",
                    "account_id": "Paccount.007",
                },
                "mpassini@example.com": {
                    "email": "mpassini@example.com",
                    "first_name": "Mina",
                    "last_name": "Passini",
                    "type": "Customer",
                    "account_id": "Paccount.017",
                },
                "dcollins@example.com": {
                    "email": "dcollins@example.com",
                    "first_name": "Deanna",
                    "last_name": "Collins",
                    "type": "Customer",
                    "account_id": "Paccount.026",
                },
                "apeterson@example.com": {
                    "email": "apeterson@example.com",
                    "first_name": "Andy",
                    "last_name": "Peterson",
                    "type": "Customer",
                    "account_id": "SDO_Paccount.030",
                },
                "indranil.sengupta@example.com": {
                    "email": "indranil.sengupta@example.com",
                    "first_name": "Indranil",
                    "last_name": "Sengupta",
                    "type": "Technician",
                    "account_id": None,
                },
                "abel.reegan@example.com": {
                    "email": "abel.reegan@example.com",
                    "first_name": "Abel",
                    "last_name": "Reegan",
                    "type": "Technician",
                    "account_id": None,
                },
            },
            "assets": {
                "Paccount.011": [
                    {
                        "name": "MacBook Air M2",
                        "product": "MacBook Air M2",
                        "serial": "D-02",
                        "purchase_date": "2025-01-02",
                        "entitlement_status": "Active",
                    },
                    {
                        "name": "Microsoft Surface Go 3",
                        "product": "Microsoft Surface Go 3",
                        "serial": "D-33",
                        "purchase_date": "2025-01-06",
                        "entitlement_status": "30 Day Warning",
                    },
                    {
                        "name": "Lenovo ThinkBook 16p",
                        "product": "Lenovo ThinkBook 16p",
                        "serial": "LP098",
                        "purchase_date": "2025-03-20",
                        "entitlement_status": "Active",
                    },
                ],
                "Paccount.024": [
                    {
                        "name": "Acer Swift 5",
                        "product": "Acer Swift 5",
                        "serial": "D-06",
                        "purchase_date": "2025-01-06",
                        "entitlement_status": "Active",
                    },
                    {
                        "name": "HP Dragonfly Pro",
                        "product": "HP Dragonfly Pro",
                        "serial": "D-37",
                        "purchase_date": "2025-02-06",
                        "entitlement_status": "Active",
                    },
                    {
                        "name": "MacBook Air M2",
                        "product": "MacBook Air M2",
                        "serial": "MB-1",
                        "purchase_date": "2025-04-02",
                        "entitlement_status": "Active",
                    },
                ],
                "Paccount.007": [
                    {
                        "name": "HP Envy 13",
                        "product": "HP Envy 13",
                        "serial": "D-13",
                        "purchase_date": "2025-01-04",
                        "entitlement_status": "Active",
                    },
                    {
                        "name": "Acer Predator Helios 300",
                        "product": "Acer Predator Helios 300",
                        "serial": "D-22",
                        "purchase_date": "2025-01-04",
                        "entitlement_status": "Active",
                    },
                    {
                        "name": "Acer Chromebook Spin 713",
                        "product": "Acer Chromebook Spin 713",
                        "serial": "D-44",
                        "purchase_date": "2026-01-04",
                        "entitlement_status": "Expired",
                    },
                ],
                "Paccount.017": [
                    {
                        "name": "HP Omen 16",
                        "product": "HP Omen 16",
                        "serial": "D-27",
                        "purchase_date": "2025-01-09",
                        "entitlement_status": "30 Day Warning",
                    },
                    {
                        "name": "MacBook Pro 14-inch",
                        "product": "MacBook Pro 14-inch",
                        "serial": "",
                        "purchase_date": "2025-02-03",
                        "entitlement_status": "Active",
                    },
                    {
                        "name": "Asus VivoBook S15",
                        "product": "Asus VivoBook S15",
                        "serial": "DL12",
                        "purchase_date": "2025-03-18",
                        "entitlement_status": "Active",
                    },
                ],
                "Paccount.026": [
                    {
                        "name": "Dell G15",
                        "product": "Dell G15",
                        "serial": "D-29",
                        "purchase_date": "2026-01-02",
                        "entitlement_status": "Expired",
                    },
                    {
                        "name": "Lenovo ThinkBook 16p",
                        "product": "Lenovo ThinkBook 16p",
                        "serial": "LP098",
                        "purchase_date": "2025-03-12",
                        "entitlement_status": "Active",
                    },
                    {
                        "name": "MacBook Pro 16-inch",
                        "product": "MacBook Pro 16-inch",
                        "serial": "MA212",
                        "purchase_date": "2025-03-25",
                        "entitlement_status": "Active",
                    },
                ],
                "SDO_Paccount.030": [
                    {
                        "name": "MSI Katana GF66",
                        "product": "MSI Katana GF66",
                        "serial": "D-30",
                        "purchase_date": "2025-01-03",
                        "entitlement_status": "30 Day Warning",
                    },
                    {
                        "name": "Lenovo Yoga 9i",
                        "product": "Lenovo Yoga 9i",
                        "serial": "",
                        "purchase_date": "2025-01-07",
                        "entitlement_status": "Active",
                    },
                    {
                        "name": "Dell Inspiron 16 Plus",
                        "product": "Dell Inspiron 16 Plus",
                        "serial": "DL12",
                        "purchase_date": "2025-04-01",
                        "entitlement_status": "Active",
                    },
                ],
            },
            "work_orders": {
                "WorkOrder_001": {
                    "id": "WorkOrder_001",
                    "asset": "Lenovo Yoga 9i",
                    "contact_email": "apeterson@example.com",
                    "territory": "San Francisco",
                    "work_type": "Laptop Troubleshooting",
                    "priority": "Low",
                    "subject": "Laptop Display Issue",
                    "description": "Facing display issue. Possible causes could be related to hardware problems such as malfunctioning internal display components, graphics driver or GPU problems, or even issues with the backlight functionality.",
                    "status": "New",
                },
                "WorkOrder_004": {
                    "id": "WorkOrder_004",
                    "asset": "MSI Katana GF66",
                    "contact_email": "apeterson@example.com",
                    "territory": "San Francisco",
                    "work_type": "Laptop Troubleshooting",
                    "priority": "Low",
                    "subject": "Laptop Performance Issue",
                    "description": "Facing performance issues such as memory errors, HDD/SSD health concerns, CPU and GPU temperature issues, excessive resource usage, malware infections, outdated drivers, BIOS/UEFI firmware, OS corruption, and critical errors in system event logs.",
                    "status": "New",
                },
                "WorkOrder_005": {
                    "id": "WorkOrder_005",
                    "asset": "MSI Katana GF66",
                    "contact_email": "apeterson@example.com",
                    "territory": "San Francisco",
                    "work_type": "Laptop Troubleshooting",
                    "priority": "Low",
                    "subject": "Laptop Performance Issue",
                    "description": "Facing performance issues such as overheating, slow performance, loud fan noise, and frequent freezing.",
                    "status": "New",
                },
            },
            "service_appointments": {
                "SA-0395": {
                    "appointment_number": "SA-0395",
                    "contact_email": "apeterson@example.com",
                    "work_order": "WorkOrder_004",
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-08T09:00:00",
                    "scheduled_end": "2025-04-08T10:00:00",
                    "resource": "Alan Reed",
                },
                "SA-0397": {
                    "appointment_number": "SA-0397",
                    "contact_email": "apeterson@example.com",
                    "work_order": "WorkOrder_005",
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-08T10:00:00",
                    "scheduled_end": "2025-04-08T11:00:00",
                    "resource": "Alan Reed",
                },
                "SA-0406": {
                    "appointment_number": "SA-0406",
                    "contact_email": "ghoskins@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-08T14:00:00",
                    "scheduled_end": "2025-04-08T15:00:00",
                    "resource": "Andy James",
                },
                "SA-0409": {
                    "appointment_number": "SA-0409",
                    "contact_email": "rjanzen@example.com",
                    "work_order": "WorkOrder_001",
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-08T11:00:00",
                    "scheduled_end": "2025-04-08T12:00:00",
                    "resource": "Janet Jones",
                },
                "SA-0412": {
                    "appointment_number": "SA-0412",
                    "contact_email": "rjanzen@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-09T09:00:00",
                    "scheduled_end": "2025-04-09T10:00:00",
                    "resource": "James Clifton",
                },
                "SA-0413": {
                    "appointment_number": "SA-0413",
                    "contact_email": "mpassini@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-09T10:00:00",
                    "scheduled_end": "2025-04-09T11:00:00",
                    "resource": "Alan Reed",
                },
                "SA-0414": {
                    "appointment_number": "SA-0414",
                    "contact_email": "rjanzen@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-09T14:00:00",
                    "scheduled_end": "2025-04-09T15:00:00",
                    "resource": "Gloria Barnes",
                },
                "SA-0416": {
                    "appointment_number": "SA-0416",
                    "contact_email": "mpassini@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "Los Angeles",
                    "scheduled_start": "2025-04-09T11:00:00",
                    "scheduled_end": "2025-04-09T12:00:00",
                    "resource": "Alan Reed",
                },
                "SA-0421": {
                    "appointment_number": "SA-0421",
                    "contact_email": "dcollins@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "Los Angeles",
                    "scheduled_start": "2025-04-09T09:00:00",
                    "scheduled_end": "2025-04-09T10:00:00",
                    "resource": "Alexander Raneri",
                },
                "SA-0422": {
                    "appointment_number": "SA-0422",
                    "contact_email": "dcollins@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "Los Angeles",
                    "scheduled_start": "2025-04-10T09:00:00",
                    "scheduled_end": "2025-04-10T10:00:00",
                    "resource": "Alexander Raneri",
                },
                "SA-0223": {
                    "appointment_number": "SA-0223",
                    "contact_email": "apeterson@example.com",
                    "work_order": "WorkOrder_001",
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-07T09:00:00",
                    "scheduled_end": "2025-04-07T10:00:00",
                    "resource": "Alan Reed",
                    "service_note": "Accessed Device Manager and checked for driver updates. Updated the keyboard driver to the latest version. Verified driver functionality after installation. Restarted the system to apply changes. Tested keyboard functionality to ensure proper operation.",
                    "subject": "Laptop keyboard driver update",
                },
                "SA-0425": {
                    "appointment_number": "SA-0425",
                    "contact_email": "rjanzen@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-10T14:00:00",
                    "scheduled_end": "2025-04-10T15:00:00",
                    "resource": "Janet Jones",
                    "service_note": "",
                    "subject": "Keyboard Issue",
                    "description": "Customer is facing keyboard issue. Troubleshooting steps tried: Disconnect and reconnect the keyboard, restart the laptop, update or reinstall the keyboard drivers, test the keyboard on another device, check the keyboard for any physical damage or debris. Issue not resolved.",
                },
                "SA-0385": {
                    "appointment_number": "SA-0385",
                    "contact_email": "mmalmin@example.com",
                    "work_order": None,
                    "status": "Scheduled",
                    "territory": "San Francisco",
                    "scheduled_start": "2025-04-08T13:00:00",
                    "scheduled_end": "2025-04-08T14:00:00",
                    "resource": "Gloria Barnes",
                    "service_note": "",
                    "subject": "HP Envy 13 Performance Issue",
                    "description": "Customer facing performance issues with their HP Envy 13 laptop. Issues may include overheating, battery not charging, laptop running slow, or software problems like corrupt OS or driver issues.",
                },
            },
            "service_resources": [
                {
                    "name": "Alan Reed",
                    "skills": ["Software", "Troubleshooting"],
                    "territory": "San Francisco",
                },
                {
                    "name": "Alexander Raneri",
                    "skills": ["Software", "Troubleshooting"],
                    "territory": "Los Angeles",
                },
                {
                    "name": "Andy James",
                    "skills": ["Software", "Troubleshooting"],
                    "territory": "San Francisco",
                },
                {
                    "name": "James Clifton",
                    "skills": ["Software", "Troubleshooting"],
                    "territory": "San Francisco",
                },
                {
                    "name": "Janet Jones",
                    "skills": ["Software", "Troubleshooting"],
                    "territory": "San Francisco",
                },
                {
                    "name": "Gloria Barnes",
                    "skills": ["Software", "Troubleshooting"],
                    "territory": "San Francisco",
                },
            ],
            "service_hours": {
                "start": 9,
                "end": 18,
                "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            },
            "available_slots": [
                "09:00-10:00",
                "10:00-11:00",
                "11:00-12:00",
                "12:00-13:00",
                "13:00-14:00",
                "14:00-15:00",
                "15:00-16:00",
                "16:00-17:00",
                "17:00-18:00",
            ],
            "kb_faq": [
                {
                    "id": "faq-001",
                    "title": "Can I schedule an appointment for multiple devices at the same time?",
                    "answer": "Yes, you can schedule a service appointment for multiple devices in a single booking. Please provide details about each device when making the appointment.",
                },
                {
                    "id": "faq-003",
                    "title": "Do I need to book an appointment, or can I walk in for service?",
                    "answer": "You can walk into our store during working hours (Monday–Friday: 9 AM–6 PM, Saturday: 10 AM–4 PM). However, we recommend scheduling an appointment online to minimize waiting times and ensure technician availability.",
                },
                {
                    "id": "faq-004",
                    "title": "Do you offer weekend or after-hours service appointments?",
                    "answer": "Yes, we offer limited weekend appointment slots on Saturdays from 10 AM–2 PM. After-hours service is available upon request and may include an additional fee.",
                },
                {
                    "id": "faq-005",
                    "title": "How long does a typical laptop repair take?",
                    "answer": "A standard laptop repair typically takes 3–5 business days, depending on the issue and part availability.",
                },
                {
                    "id": "faq-006",
                    "title": "Will I receive a confirmation and reminder for my scheduled service appointment?",
                    "answer": "Yes, once you schedule an appointment, you will receive an email and SMS confirmation. A reminder will also be sent 24 hours before your appointment.",
                },
                {
                    "id": "faq-007",
                    "title": "Can I book a service appointment online or do I need to call or any other way?",
                    "answer": "You can book your appointment online through our website, by calling our customer service line, or visit our store in person.",
                },
                {
                    "id": "faq-008",
                    "title": "Do I need to provide my laptop details before scheduling the appointment?",
                    "answer": "Yes, providing your laptop's brand, model, and issue details helps our technicians prepare efficiently for your service.",
                },
                {
                    "id": "faq-009",
                    "title": "Can I choose a specific technician while scheduling my repair appointment?",
                    "answer": "No, technician assignment is based on availability. Our system automatically assigns the best available technician to ensure timely service.",
                },
                {
                    "id": "faq-010",
                    "title": "How long does a standard laptop repair take on average?",
                    "answer": "A standard laptop repair typically takes 1 to 3 business days, depending on the complexity of the issue and the availability of replacement parts.",
                },
                {
                    "id": "faq-011",
                    "title": "What is the maximum number of appointments a technician can handle in a day?",
                    "answer": "A technician can typically handle 6–8 repair appointments per day, depending on the complexity of the repairs.",
                },
                {
                    "id": "faq-013",
                    "title": "Is the Apple Magic Keyboard compatible with MacBook Air?",
                    "answer": "Yes, the Apple Magic Keyboard is compatible with MacBook Air models supporting Bluetooth connectivity.",
                },
                {
                    "id": "faq-014",
                    "title": "Is the Apple MagSafe Charger compatible with iPhone 13?",
                    "answer": "Yes, the Apple MagSafe Charger is fully compatible with the iPhone 13 and other MagSafe-supported iPhones.",
                },
                {
                    "id": "faq-015",
                    "title": "Is the Apple AirPods Max compatible with MacBook Pro?",
                    "answer": "Yes, the Apple AirPods Max can be paired with MacBook Pro via Bluetooth for high-fidelity audio and noise cancellation.",
                },
                {
                    "id": "faq-016",
                    "title": "Is the Apple MagSafe Charger compatible with MacBook Air?",
                    "answer": "No, the Apple MagSafe Charger is designed for iPhones. MacBook Air requires a USB-C Power Adapter.",
                },
                {
                    "id": "faq-017",
                    "title": "Is the Apple AirPods Pro compatible with iPhone 16 Pro?",
                    "answer": "Yes, the Apple AirPods Pro is compatible with the iPhone 16 Pro and all iPhones with Bluetooth.",
                },
                {
                    "id": "faq-018",
                    "title": "Is the Apple Magic Mouse compatible with MacBook Pro?",
                    "answer": "Yes, the Apple Magic Mouse is compatible with MacBook Pro models that support Bluetooth.",
                },
                {
                    "id": "faq-019",
                    "title": "What is the estimated repair time for a standard mobile?",
                    "answer": "Most standard mobile phone repairs (such as battery replacement, screen repair) take 1–2 hours, while more complex issues may take up to 48 hours.",
                },
            ],
            "kb_troubleshooting": [
                {
                    "id": "ts-001",
                    "title": "Laptop Not Turning On",
                    "steps": [
                        "Check the power adapter and battery connection.",
                        "Try a different power outlet.",
                        "Remove the battery (if removable) and reconnect it.",
                        "Perform a hard reset (hold the power button for 30 seconds).",
                        "Check for any indicator lights or beeping sounds that suggest hardware failure.",
                    ],
                },
                {
                    "id": "ts-002",
                    "title": "Slow Performance / Freezing",
                    "steps": [
                        "Check for background applications consuming high CPU/memory.",
                        "Scan for malware/viruses.",
                        "Clear temporary files and update drivers.",
                        "Upgrade RAM or replace the hard drive with an SSD if needed.",
                    ],
                },
                {
                    "id": "ts-003",
                    "title": "Overheating / Fan Noise",
                    "steps": [
                        "Clean dust from vents and fans.",
                        "Check if any software is overloading the system.",
                        "Update BIOS and drivers.",
                        "Apply thermal paste if necessary.",
                    ],
                },
                {
                    "id": "ts-004",
                    "title": "Display Issues (Blank Screen, Flickering)",
                    "steps": [
                        "Check if an external monitor works.",
                        "Update or roll back the display driver.",
                        "Perform a power cycle.",
                        "Inspect the display cable for damage.",
                    ],
                },
                {
                    "id": "ts-005",
                    "title": "Laptop Charger Troubleshooting",
                    "steps": [
                        "Check the power source and ensure it's working.",
                        "Inspect the charger and cable for any visible damage.",
                        "Ensure the charger is securely connected to the laptop.",
                        "Check the battery status in the system settings.",
                        "Update the power management and battery drivers.",
                    ],
                },
            ],
            "technician_troubleshooting_guides": {
                "power_failure": {
                    "title": "Laptop Power Failure Troubleshooting",
                    "sections": [
                        {
                            "name": "External Power Supply Verification",
                            "steps": [
                                "Use a multimeter to measure the output voltage and amperage of the AC adapter.",
                                "Visually inspect the DC jack on the laptop for physical damage.",
                                "Use a continuity tester to check the DC jack for continuity.",
                                "Test the laptop with a known good AC adapter.",
                            ],
                        },
                        {
                            "name": "Internal Power Checks",
                            "steps": [
                                "Open the laptop casing and use a multimeter to verify correct voltage on the motherboard's main power rails.",
                                "Inspect the motherboard for visible signs of damage.",
                                "Check the CMOS battery voltage.",
                            ],
                        },
                        {
                            "name": "Component Isolation",
                            "steps": [
                                "Disconnect any non-essential peripherals.",
                                "Test with only one RAM module installed at a time.",
                                "Inspect the power button assembly.",
                                "Check the fuse on the motherboard with a multimeter.",
                            ],
                        },
                    ],
                },
                "display_malfunction": {
                    "title": "Laptop Display Malfunction Diagnosis",
                    "sections": [
                        {
                            "name": "External Display Test",
                            "steps": [
                                "Connect the laptop to an external monitor.",
                                "If the external monitor displays correctly, the issue is with the internal display.",
                                "Check the external display cable for damage.",
                            ],
                        },
                        {
                            "name": "Internal Display Checks",
                            "steps": [
                                "Disassemble the laptop to access the LCD/LED panel connections.",
                                "Inspect the LCD/LED panel connections for loose or damaged cables.",
                                "Test the backlight functionality by shining a flashlight on the screen.",
                                "Check the LVDS/eDP cable for damage.",
                            ],
                        },
                        {
                            "name": "Graphics Driver and GPU",
                            "steps": [
                                "Update the graphics drivers to the latest version.",
                                "Run a GPU stress test.",
                                "Reseat the RAM modules.",
                            ],
                        },
                    ],
                },
                "bluetooth_problems": {
                    "title": "Laptop Bluetooth Problems Resolution",
                    "sections": [
                        {
                            "name": "Hardware Verification",
                            "steps": [
                                "Verify that the Bluetooth module is present and properly connected.",
                                "Check the antenna connections for the Bluetooth module.",
                                "Inspect the Bluetooth module for physical damage.",
                            ],
                        },
                        {
                            "name": "Driver and Software",
                            "steps": [
                                "Update the Bluetooth drivers to the latest version.",
                                "Check the status of the Bluetooth service in Windows Services.",
                                "Review the Device Manager for errors.",
                            ],
                        },
                        {
                            "name": "Interference and Environment",
                            "steps": [
                                "Check for wireless interference from other devices.",
                                "Test with different Bluetooth devices.",
                                "Test in a different physical location.",
                            ],
                        },
                    ],
                },
                "performance_degradation": {
                    "title": "Laptop Performance Degradation Analysis",
                    "sections": [
                        {
                            "name": "Hardware Diagnostics",
                            "steps": [
                                "Run a comprehensive memory diagnostic test (e.g., Memtest86+).",
                                "Use a diagnostic tool (e.g., CrystalDiskInfo) to check the health of the HDD/SSD.",
                                "Use a stress test utility (e.g., Prime95, FurMark) to monitor CPU and GPU temperatures.",
                            ],
                        },
                        {
                            "name": "Software Optimization",
                            "steps": [
                                "Open Task Manager or Resource Monitor to analyze system resource usage.",
                                "Perform a full system scan for malware.",
                                "Review startup programs and services. Disable unnecessary items.",
                            ],
                        },
                        {
                            "name": "Driver and OS Review",
                            "steps": [
                                "Update the laptop's BIOS/UEFI firmware.",
                                "Update all system drivers.",
                                "Run the System File Checker (SFC) and DISM tools.",
                                "Review system event logs for critical errors.",
                            ],
                        },
                    ],
                },
                "charging_failure": {
                    "title": "Laptop Charging Failure Procedure",
                    "sections": [
                        {
                            "name": "Charging Circuit",
                            "steps": [
                                "Use a multimeter to test the charging IC.",
                                "Check for short circuits on the charging circuit.",
                                "Verify the functionality of the MOSFETs.",
                            ],
                        },
                        {
                            "name": "Battery Diagnostics",
                            "steps": [
                                "Use a multimeter to test the battery voltage and health.",
                                "Check the battery connector for damage or corrosion.",
                                "Test the battery with a known good battery tester.",
                            ],
                        },
                        {
                            "name": "Power Path Tracing",
                            "steps": [
                                "Trace the power path from the DC jack to the battery connector.",
                                "Identify and replace any faulty components.",
                                "Check the fuse on the motherboard for continuity.",
                            ],
                        },
                    ],
                },
                "overheating": {
                    "title": "Laptop Overheating Troubleshooting",
                    "sections": [
                        {
                            "name": "Cooling System Inspection",
                            "steps": [
                                "Disassemble the laptop to access the heat sink and fan assembly.",
                                "Clean the heat sink and fan assembly.",
                                "Verify that the fan is spinning freely and at the correct RPM.",
                                "Remove old thermal paste and reapply a thin, even layer.",
                            ],
                        },
                        {
                            "name": "Airflow Assessment",
                            "steps": [
                                "Check for blocked vents and airflow restrictions.",
                                "Ensure internal airflow is not obstructed by cables.",
                                "Measure the ambient air temperature.",
                            ],
                        },
                        {
                            "name": "Sensor and Monitoring",
                            "steps": [
                                "Use monitoring software to track CPU/GPU temperatures under load.",
                                "Check thermal sensor readings for accuracy.",
                                "Review power consumption to identify components drawing excessive power.",
                            ],
                        },
                    ],
                },
                "keyboard": {
                    "title": "Laptop Keyboard Troubleshooting",
                    "sections": [
                        {
                            "name": "Initial Checks",
                            "steps": [
                                "Disconnect and reconnect the keyboard.",
                                "Restart the laptop.",
                                "Update or reinstall the keyboard drivers.",
                                "Test the keyboard on another device.",
                                "Check for physical damage or debris.",
                            ],
                        },
                    ],
                },
            },
            "technician_checklists": {
                "keyboard": [
                    "Gather Necessary Tools & Equipment: Replacement keyboard, screwdrivers, anti-static wrist strap.",
                    "Software & Diagnostic Tools: External keyboard, on-screen keyboard, BIOS keyboard test.",
                    "Document Service Details: Record issue description, previous attempts to fix.",
                    "Communicate with Customer: Confirm issue details, expected resolution time.",
                ],
                "display": [
                    "Gather Necessary Tools & Equipment: External monitor, cables, display replacement kit.",
                    "Software & Diagnostic Tools: Check brightness settings, driver updates, external display test.",
                    "Document Service Details: Note display behavior, error messages, external display results.",
                    "Communicate with Customer: Explain findings, potential solutions, estimated repair time.",
                ],
                "power_charging": [
                    "Gather Necessary Tools & Equipment: Multimeter, replacement charger, power jack tester.",
                    "Software & Diagnostic Tools: Battery health check, BIOS power settings, power cycle test.",
                    "Document Service Details: Battery age, charging patterns, previous repair history.",
                    "Communicate with Customer: Inform about battery condition, power component status.",
                ],
                "slow_performance": [
                    "Gather Necessary Tools & Equipment: RAM upgrade kit, SSD replacement tools.",
                    "Software & Diagnostic Tools: Task Manager analysis, disk cleanup, startup program review.",
                    "Document Service Details: Performance logs, previous optimization attempts.",
                    "Communicate with Customer: Explain potential causes, performance improvement options.",
                ],
                "overheating": [
                    "Gather Necessary Tools & Equipment: Thermal paste, cooling pad, compressed air.",
                    "Software & Diagnostic Tools: Temperature monitoring software, fan speed test.",
                    "Document Service Details: Temperature logs, ventilation condition.",
                    "Communicate with Customer: Advise on cleaning, ventilation improvement steps.",
                ],
                "hard_drive_os": [
                    "Gather Necessary Tools & Equipment: External drive, OS installation media, SATA cables.",
                    "Software & Diagnostic Tools: Disk health check, OS repair tools, boot recovery.",
                    "Document Service Details: Error messages, OS version, file backup status.",
                    "Communicate with Customer: Inform about data backup needs, recovery options.",
                ],
            },
            "scheduling_actions": [],
            "created_work_orders": [],
        }

    def _find_user(self, email: str) -> dict | None:
        return self.data["users"].get(email)

    def _get_appointments_for_user(self, email: str) -> list[dict]:
        return [
            appt
            for appt in self.data["service_appointments"].values()
            if appt["contact_email"] == email
        ]

    def _is_within_service_hours(self, hour: int) -> bool:
        hours = self.data["service_hours"]
        return hours["start"] <= hour < hours["end"]

    def _match_troubleshooting_guide(self, query: str) -> dict | None:
        query_lower = query.lower()
        keyword_map = {
            "power_failure": [
                "power failure",
                "won't power on",
                "not turning on",
                "wont power",
            ],
            "display_malfunction": [
                "display",
                "screen",
                "blank screen",
                "flickering",
                "internal display",
            ],
            "bluetooth_problems": ["bluetooth"],
            "performance_degradation": [
                "performance",
                "slow",
                "degraded",
                "freezing",
                "memory",
                "cpu",
                "ram failure",
                "hard drive",
            ],
            "charging_failure": ["charging", "charger", "battery", "not charging"],
            "overheating": [
                "overheating",
                "overheat",
                "thermal",
                "heating",
                "fan noise",
                "airflow",
                "cooling",
            ],
            "keyboard": ["keyboard"],
        }
        for key, keywords in keyword_map.items():
            if any(kw in query_lower for kw in keywords):
                return self.data["technician_troubleshooting_guides"].get(key)
        return None

    def register_tools(self, registry: ToolRegistry) -> None:
        env = self

        @registry.tool(
            "Search the FAQ knowledge base for information about appointments, repair times, compatibility, and service policies"
        )
        def FSA_Knowledge_FAQ(query: str) -> str:
            query_lower = query.lower()
            results = []
            for article in env.data["kb_faq"]:
                if (
                    query_lower in article["title"].lower()
                    or any(
                        word in article["title"].lower()
                        for word in query_lower.split()
                        if len(word) > 3
                    )
                    or any(
                        word in article["answer"].lower()
                        for word in query_lower.split()
                        if len(word) > 3
                    )
                ):
                    results.append(
                        {
                            "id": article["id"],
                            "title": article["title"],
                            "answer": article["answer"],
                        }
                    )

            if not results:
                results = env.data["kb_faq"][:3]
            return json.dumps({"results": results[:5]})

        @registry.tool(
            "Identify user type (Customer or Technician) based on email address"
        )
        def FSA_Identify_User_Type_Based_on_Email(email: str) -> str:
            user = env._find_user(email)
            if not user:
                return json.dumps({"error": f"User with email {email} not found"})
            return json.dumps(
                {
                    "email": user["email"],
                    "name": f"{user['first_name']} {user['last_name']}",
                    "user_type": user["type"],
                }
            )

        @registry.tool(
            "Get troubleshooting steps from the technician knowledge library based on the issue description"
        )
        def FSA_Technician_Get_Troubleshooting_Steps(issue_description: str) -> str:
            guide = env._match_troubleshooting_guide(issue_description)
            if not guide:
                return json.dumps(
                    {
                        "message": "No specific troubleshooting steps found for this issue in the uploaded files."
                    }
                )
            sections = []
            for section in guide["sections"]:
                sections.append(
                    {
                        "name": section["name"],
                        "steps": section["steps"],
                    }
                )
            return json.dumps({"title": guide["title"], "sections": sections})

        @registry.tool(
            "Get appointment details for a user based on their email and optional date/time preference"
        )
        def FSA_Get_Appointment_details_based_on_User_Input(
            email: str, preferred_date: str = "", preferred_time: str = ""
        ) -> str:
            # preferred_date / preferred_time are part of the real tool's
            # contract (the agent is graded on passing them when the user states
            # a preference), but this mock returns the user's full appointment
            # list unfiltered — matching the fixed fixtures the tasks expect.
            user = env._find_user(email)
            if not user:
                return json.dumps({"error": f"User with email {email} not found"})
            appointments = env._get_appointments_for_user(email)
            if not appointments:
                return json.dumps({"error": "No appointments found for this user"})
            appt_list = []
            for appt in appointments:
                appt_list.append(
                    {
                        "appointment_number": appt["appointment_number"],
                        "status": appt["status"],
                        "scheduled_start": appt["scheduled_start"],
                        "scheduled_end": appt["scheduled_end"],
                        "territory": appt["territory"],
                        "resource": appt["resource"],
                    }
                )
            return json.dumps(
                {
                    "appointments": appt_list,
                    "message": "Please select the appointment you want to reschedule.",
                }
            )

        @registry.tool(
            "Unschedule an existing service appointment to allow rescheduling"
        )
        def FSA_Unschedule_Appointment(appointment_number: str) -> str:
            appt = env.data["service_appointments"].get(appointment_number)
            if not appt:
                return json.dumps(
                    {"error": f"Appointment {appointment_number} not found"}
                )
            appt["status"] = "Unscheduled"
            env.data["scheduling_actions"].append(
                {"action": "unschedule", "appointment": appointment_number}
            )
            return json.dumps(
                {
                    "status": "Success",
                    "appointment_number": appointment_number,
                    "message": "Appointment unscheduled successfully",
                }
            )

        @registry.tool("Get available time slots for rescheduling an appointment")
        def FSA_Get_Available_Slots_Based_on_Appointment_Number(
            appointment_number: str, preferred_date: str = ""
        ) -> str:
            # preferred_date is accepted for tool-contract fidelity (and arg
            # grading) but the mock returns the fixed slot list regardless.
            appt = env.data["service_appointments"].get(appointment_number)
            if not appt:
                return json.dumps(
                    {"error": f"Appointment {appointment_number} not found"}
                )
            slots = env.data["available_slots"]
            return json.dumps(
                {
                    "appointment_number": appointment_number,
                    "available_slots": slots,
                    "message": "Please select a slot number from the list above.",
                }
            )

        @registry.tool("Update the arrival window for a service appointment")
        def FSA_Update_Arrival_Windows(
            appointment_number: str, start_time: str, end_time: str
        ) -> str:
            appt = env.data["service_appointments"].get(appointment_number)
            if not appt:
                return json.dumps(
                    {"error": f"Appointment {appointment_number} not found"}
                )
            appt["scheduled_start"] = start_time
            appt["scheduled_end"] = end_time
            env.data["scheduling_actions"].append(
                {
                    "action": "update_arrival",
                    "appointment": appointment_number,
                    "start": start_time,
                    "end": end_time,
                }
            )
            return json.dumps(
                {
                    "status": "Success",
                    "appointment_number": appointment_number,
                    "arrival_window": f"{start_time} - {end_time}",
                }
            )

        @registry.tool(
            "Schedule or reschedule a service appointment to a specific time slot"
        )
        def FSA_Schedule_Appointment(appointment_number: str, slot: str = "") -> str:
            appt = env.data["service_appointments"].get(appointment_number)
            if not appt:
                return json.dumps(
                    {"error": f"Appointment {appointment_number} not found"}
                )
            appt["status"] = "Scheduled"
            env.data["scheduling_actions"].append(
                {"action": "schedule", "appointment": appointment_number, "slot": slot}
            )
            return json.dumps(
                {
                    "status": "Success",
                    "appointment_number": appointment_number,
                    "message": f"Appointment {appointment_number} has been successfully scheduled.",
                    "resource": appt["resource"],
                }
            )

        @registry.tool(
            "Identify the issue based on work order details and customer description"
        )
        def Field_Service_Agent_Identify_the_issue(
            email: str, asset_name: str = "", issue_description: str = ""
        ) -> str:
            user = env._find_user(email)
            if not user:
                return json.dumps({"error": f"User with email {email} not found"})
            account_id = user.get("account_id")
            assets = env.data["assets"].get(account_id, [])
            matched_asset = None
            if asset_name:
                for asset in assets:
                    if (
                        asset_name.lower() in asset["name"].lower()
                        or asset["name"].lower() in asset_name.lower()
                    ):
                        matched_asset = asset
                        break
            if not matched_asset and assets:
                return json.dumps(
                    {
                        "error": f"Asset '{asset_name}' not found for this customer",
                        "available_assets": [a["name"] for a in assets],
                    }
                )
            issue_category = "General"
            desc_lower = (issue_description or "").lower()
            if any(
                w in desc_lower for w in ["display", "screen", "blank", "flickering"]
            ):
                issue_category = "Display"
            elif any(w in desc_lower for w in ["performance", "slow", "freezing"]):
                issue_category = "Performance"
            elif any(w in desc_lower for w in ["charging", "power", "battery"]):
                issue_category = "Power/Charging"
            elif any(w in desc_lower for w in ["keyboard"]):
                issue_category = "Keyboard"
            elif any(w in desc_lower for w in ["sound", "audio"]):
                issue_category = "Sound"
            elif any(w in desc_lower for w in ["overheat", "thermal", "hot"]):
                issue_category = "Overheating"

            return json.dumps(
                {
                    "customer": f"{user['first_name']} {user['last_name']}",
                    "asset": matched_asset["name"] if matched_asset else None,
                    "issue_category": issue_category,
                    "issue_description": issue_description,
                }
            )

        @registry.tool("Get warranty and entitlement details for a customer's asset")
        def Get_Warranty_Details_Field_Service_Agent(
            email: str, asset_name: str = ""
        ) -> str:
            user = env._find_user(email)
            if not user:
                return json.dumps({"error": f"User with email {email} not found"})
            account_id = user.get("account_id")
            assets = env.data["assets"].get(account_id, [])
            if asset_name:
                for asset in assets:
                    if (
                        asset_name.lower() in asset["name"].lower()
                        or asset["name"].lower() in asset_name.lower()
                    ):
                        return json.dumps(
                            {
                                "asset": asset["name"],
                                "serial_number": asset["serial"],
                                "purchase_date": asset["purchase_date"],
                                "entitlement_status": asset["entitlement_status"],
                            }
                        )
                return json.dumps(
                    {
                        "error": f"Asset '{asset_name}' not found",
                        "available_assets": [a["name"] for a in assets],
                    }
                )
            return json.dumps(
                {
                    "assets": [
                        {
                            "name": a["name"],
                            "entitlement_status": a["entitlement_status"],
                            "purchase_date": a["purchase_date"],
                        }
                        for a in assets
                    ]
                }
            )

        @registry.tool(
            "Create a new work order for a customer's issue and return available appointment slots"
        )
        def Get_New_Work_Order_by_User_Input(
            email: str,
            asset_name: str = "",
            issue_description: str = "",
            work_type: str = "Laptop Troubleshooting",
        ) -> str:
            user = env._find_user(email)
            if not user:
                return json.dumps({"error": f"User with email {email} not found"})
            wo_id = f"WorkOrder_{random.randint(100, 999)}"
            wo = {
                "id": wo_id,
                "asset": asset_name,
                "contact_email": email,
                "territory": "San Francisco",
                "work_type": work_type,
                "priority": "Low",
                "subject": issue_description[:50]
                if issue_description
                else "Service Request",
                "description": issue_description,
                "status": "New",
            }
            env.data["created_work_orders"].append(wo)
            appt_num = _appointment_number()
            env.data["service_appointments"][appt_num] = {
                "appointment_number": appt_num,
                "contact_email": email,
                "work_order": wo_id,
                "status": "None",
                "territory": "San Francisco",
                "scheduled_start": None,
                "scheduled_end": None,
                "resource": None,
            }
            return json.dumps(
                {
                    "work_order_id": wo_id,
                    "appointment_number": appt_num,
                    "available_slots": env.data["available_slots"],
                    "message": "Work order created. Please select an available slot.",
                }
            )

        @registry.tool(
            "Get the pre-work brief for a technician including work order details, service notes, and checklist"
        )
        def FSA_Get_Pre_Work_Brief_for_Technician(appointment_number: str) -> str:
            appt = env.data["service_appointments"].get(appointment_number)
            if not appt:
                return json.dumps(
                    {"error": f"Appointment {appointment_number} not found"}
                )

            brief = {
                "appointment_number": appointment_number,
                "status": appt["status"],
                "scheduled_start": appt.get("scheduled_start"),
                "resource": appt.get("resource"),
            }

            subject = appt.get("subject", "")
            description = appt.get("description", "")
            service_note = appt.get("service_note", "")

            if appt.get("work_order") and appt["work_order"] in env.data["work_orders"]:
                wo = env.data["work_orders"][appt["work_order"]]
                brief["work_order"] = {
                    "subject": wo["subject"],
                    "description": wo["description"],
                    "status": wo["status"],
                }
                subject = subject or wo["subject"]
                description = description or wo["description"]

            if service_note:
                brief["service_note"] = service_note

            checklist_key = None
            combined = (subject + " " + description).lower()
            if "keyboard" in combined:
                checklist_key = "keyboard"
            elif "display" in combined:
                checklist_key = "display"
            elif "power" in combined or "charging" in combined:
                checklist_key = "power_charging"
            elif "performance" in combined or "slow" in combined:
                checklist_key = "slow_performance"
            elif "overheat" in combined or "thermal" in combined:
                checklist_key = "overheating"
            elif "hard drive" in combined or "os" in combined:
                checklist_key = "hard_drive_os"

            if checklist_key:
                brief["checklist"] = env.data["technician_checklists"][checklist_key]

            return json.dumps(brief)
