# Ported from anthropics/anthropic-cookbook (threat_intel_enrichment_agent), MIT. Reimplemented as pure-Python ToolEnvironment (no upstream code copied).
"""Security / threat-intel stateful tool domain.

Provides tools that read/write a shared state dict simulating a SOC threat
intelligence enrichment workflow: IP reputation, domain reputation, file
hash (malware) reputation, CVE records, MITRE ATT&CK technique lookups, an
internal allowlist of trusted indicators, an indicator blocklist, and an
incident triage/close workflow.
"""

from __future__ import annotations

import json
import random
import string

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry

_VALID_SEVERITIES = ("low", "medium", "high", "critical")


def _incident_id() -> str:
    return "INC-" + "".join(random.choices(string.digits, k=4))


@register_domain("security")
class SecurityEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "ip_reputation": {
                "203.0.113.55": {
                    "ip": "203.0.113.55",
                    "country": "RU",
                    "city": "Moscow",
                    "asn": "AS12389",
                    "isp": "SomeNet LLC",
                    "abuse_confidence_score": 92,
                    "total_reports": 147,
                    "first_seen": "2024-11-02",
                    "last_reported": "2026-09-08",
                    "threat_types": ["botnet", "scanning"],
                    "known_malware_associations": ["Emotet"],
                    "tags": ["tor_exit"],
                    "open_ports": [22, 443, 8080],
                    "is_tor_exit_node": True,
                    "is_known_proxy": False,
                },
                "198.51.100.23": {
                    "ip": "198.51.100.23",
                    "country": "NL",
                    "city": "Amsterdam",
                    "asn": "AS60781",
                    "isp": "Leaseweb",
                    "abuse_confidence_score": 78,
                    "total_reports": 61,
                    "first_seen": "2025-06-14",
                    "last_reported": "2026-09-05",
                    "threat_types": ["c2", "malware_hosting"],
                    "known_malware_associations": ["QakBot"],
                    "tags": ["known_proxy"],
                    "open_ports": [443],
                    "is_tor_exit_node": False,
                    "is_known_proxy": True,
                },
                "8.8.8.8": {
                    "ip": "8.8.8.8",
                    "country": "US",
                    "city": "Mountain View",
                    "asn": "AS15169",
                    "isp": "Google LLC",
                    "abuse_confidence_score": 0,
                    "total_reports": 0,
                    "first_seen": "2010-01-01",
                    "last_reported": None,
                    "threat_types": [],
                    "known_malware_associations": [],
                    "tags": ["public_dns"],
                    "open_ports": [53],
                    "is_tor_exit_node": False,
                    "is_known_proxy": False,
                },
                "10.0.0.5": {
                    "ip": "10.0.0.5",
                    "country": "US",
                    "city": "Internal",
                    "asn": "N/A",
                    "isp": "Corporate Network",
                    "abuse_confidence_score": 0,
                    "total_reports": 0,
                    "first_seen": "2019-03-01",
                    "last_reported": None,
                    "threat_types": [],
                    "known_malware_associations": [],
                    "tags": ["internal", "corp_vpn"],
                    "open_ports": [443],
                    "is_tor_exit_node": False,
                    "is_known_proxy": False,
                },
                "45.33.32.156": {
                    "ip": "45.33.32.156",
                    "country": "US",
                    "city": "Fremont",
                    "asn": "AS63949",
                    "isp": "Linode",
                    "abuse_confidence_score": 55,
                    "total_reports": 19,
                    "first_seen": "2026-02-11",
                    "last_reported": "2026-08-30",
                    "threat_types": ["scanning"],
                    "known_malware_associations": [],
                    "tags": [],
                    "open_ports": [22, 80, 443],
                    "is_tor_exit_node": False,
                    "is_known_proxy": False,
                },
            },
            "domains": {
                "malicious-update-service.com": {
                    "domain": "malicious-update-service.com",
                    "registrar": "NameCheap",
                    "registrant_country": "PA",
                    "hosting_provider": "Leaseweb",
                    "hosting_country": "NL",
                    "reputation_score": 4,
                    "category": "malware",
                    "subcategory": "fake_updater",
                    "ssl_issuer": "Let's Encrypt",
                    "active": True,
                    "registration_date": "2026-05-20",
                    "ssl_valid_from": "2026-05-21",
                    "ip_addresses": ["198.51.100.23"],
                    "tags": ["typosquat"],
                    "associated_malware": ["QakBot"],
                    "dns_records": {
                        "A": ["198.51.100.23"],
                        "NS": ["ns1.badreg.net"],
                        "TXT": [],
                    },
                },
                "clean-cdn-assets.net": {
                    "domain": "clean-cdn-assets.net",
                    "registrar": "GoDaddy",
                    "registrant_country": "US",
                    "hosting_provider": "Cloudflare",
                    "hosting_country": "US",
                    "reputation_score": 96,
                    "category": "content_delivery",
                    "subcategory": "cdn",
                    "ssl_issuer": "DigiCert",
                    "active": True,
                    "registration_date": "2015-08-02",
                    "ssl_valid_from": "2026-01-10",
                    "ip_addresses": ["104.16.85.20"],
                    "tags": [],
                    "associated_malware": [],
                    "dns_records": {
                        "A": ["104.16.85.20"],
                        "NS": ["ns1.cloudflare.com"],
                        "TXT": ["v=spf1 include:_spf.example.com ~all"],
                    },
                },
                "corp-intranet.internal": {
                    "domain": "corp-intranet.internal",
                    "registrar": "Internal",
                    "registrant_country": "US",
                    "hosting_provider": "Corporate Network",
                    "hosting_country": "US",
                    "reputation_score": 100,
                    "category": "internal",
                    "subcategory": "corporate",
                    "ssl_issuer": "Internal CA",
                    "active": True,
                    "registration_date": "2018-01-01",
                    "ssl_valid_from": "2026-01-01",
                    "ip_addresses": ["10.0.0.8"],
                    "tags": ["internal"],
                    "associated_malware": [],
                    "dns_records": {"A": ["10.0.0.8"], "NS": [], "TXT": []},
                },
            },
            "file_hashes": {
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b85": {
                    "hash": (
                        "e3b0c44298fc1c149afbf4c8996fb92427ae41e"
                        "4649b934ca495991b7852b85"
                    ),
                    "hash_type": "sha256",
                    "file_name": "invoice_update.exe",
                    "file_type": "PE32 executable",
                    "detections": 58,
                    "total_engines": 70,
                    "file_size_bytes": 481280,
                    "detection_rate": "82.9%",
                    "malware_family": "Emotet",
                    "malware_type": "trojan",
                    "severity": "critical",
                    "first_seen": "2026-04-02",
                    "last_seen": "2026-09-01",
                    "tags": ["banking_trojan", "loader"],
                    "contacted_ips": ["203.0.113.55"],
                    "contacted_domains": ["malicious-update-service.com"],
                    "behavior_summary": (
                        "Drops secondary payload, establishes persistence via "
                        "registry run key, exfiltrates browser credentials."
                    ),
                },
                "44d88612fea8a8f36de82e1278abb02f": {
                    "hash": "44d88612fea8a8f36de82e1278abb02f",
                    "hash_type": "md5",
                    "file_name": "eicar.com",
                    "file_type": "text",
                    "detections": 68,
                    "total_engines": 70,
                    "file_size_bytes": 68,
                    "detection_rate": "97.1%",
                    "malware_family": "EICAR-Test-File",
                    "malware_type": "test",
                    "severity": "low",
                    "first_seen": "2003-01-01",
                    "last_seen": "2026-08-15",
                    "tags": ["antivirus_test_file"],
                    "contacted_ips": [],
                    "contacted_domains": [],
                    "behavior_summary": "Industry-standard AV test string, not malicious.",
                },
                "5f4dcc3b5aa765d61d8327deb882cf99aabbccddeeff00112233445566778899": {
                    "hash": (
                        "5f4dcc3b5aa765d61d8327deb882cf99aabbccdd"
                        "eeff00112233445566778899"
                    ),
                    "hash_type": "sha256",
                    "file_name": "system_update.dll",
                    "file_type": "PE32 DLL",
                    "detections": 2,
                    "total_engines": 70,
                    "file_size_bytes": 102400,
                    "detection_rate": "2.9%",
                    "malware_family": None,
                    "malware_type": "unknown",
                    "severity": "informational",
                    "first_seen": "2026-07-19",
                    "last_seen": "2026-07-19",
                    "tags": ["low_prevalence"],
                    "contacted_ips": [],
                    "contacted_domains": [],
                    "behavior_summary": "Low detection rate; likely a benign or packed binary.",
                },
            },
            "cves": {
                "CVE-2024-3094": {
                    "cve_id": "CVE-2024-3094",
                    "description": (
                        "Malicious code embedded in xz-utils build scripts allows "
                        "remote code execution via backdoored sshd."
                    ),
                    "severity": "Critical",
                    "cvss_score": 10.0,
                    "affected_products": ["xz-utils 5.6.0", "xz-utils 5.6.1"],
                    "exploited_in_wild": True,
                    "published_date": "2024-03-29",
                },
                "CVE-2023-44487": {
                    "cve_id": "CVE-2023-44487",
                    "description": (
                        "HTTP/2 Rapid Reset allows attackers to cause denial of "
                        "service via rapid stream cancellation."
                    ),
                    "severity": "High",
                    "cvss_score": 7.5,
                    "affected_products": ["nginx", "Apache HTTP Server", "envoy"],
                    "exploited_in_wild": True,
                    "published_date": "2023-10-10",
                },
            },
            "mitre_techniques": {
                "T1059": {
                    "id": "T1059",
                    "name": "Command and Scripting Interpreter",
                    "tactic": "Execution",
                    "description": (
                        "Adversaries may abuse command and script interpreters to "
                        "execute commands, scripts, or binaries."
                    ),
                    "associated_groups": ["APT29", "Lazarus Group"],
                    "detection_suggestions": [
                        "Monitor process execution of cmd.exe, powershell.exe",
                        "Enable script block logging",
                    ],
                },
                "T1071": {
                    "id": "T1071",
                    "name": "Application Layer Protocol",
                    "tactic": "Command and Control",
                    "description": (
                        "Adversaries may communicate using OSI application layer "
                        "protocols to blend in with existing traffic."
                    ),
                    "associated_groups": ["FIN7", "QakBot operators"],
                    "detection_suggestions": [
                        "Inspect DNS and HTTP traffic for beaconing patterns",
                        "Baseline typical outbound protocol usage",
                    ],
                },
            },
            "allowlist": {
                "10.0.0.5": {"type": "ip", "reason": "Corporate VPN gateway"},
                "10.0.0.8": {"type": "ip", "reason": "Internal DNS server"},
                "corp-intranet.internal": {
                    "type": "domain",
                    "reason": "Internal corporate domain",
                },
            },
            "blocklist": {},
            "incidents": {
                "INC-1001": {
                    "incident_id": "INC-1001",
                    "title": "Suspicious outbound traffic to known C2 IP",
                    "indicator": "203.0.113.55",
                    "indicator_type": "ip",
                    "severity": "high",
                    "status": "Open",
                    "notes": [],
                    "created_at": "2026-09-09",
                    "triaged_at": None,
                    "closed_at": None,
                    "resolution": None,
                },
                "INC-1002": {
                    "incident_id": "INC-1002",
                    "title": "Malware download detected from typosquat domain",
                    "indicator": "malicious-update-service.com",
                    "indicator_type": "domain",
                    "severity": "critical",
                    "status": "Triaged",
                    "notes": ["Confirmed Emotet loader via sandbox detonation"],
                    "created_at": "2026-09-07",
                    "triaged_at": "2026-09-08",
                    "closed_at": None,
                    "resolution": None,
                },
            },
        }

    def register_tools(self, registry: ToolRegistry) -> None:
        data = self.data

        @registry.tool("Look up reputation and threat intel for an IP address")
        def lookup_ip_reputation(ip_address: str) -> str:
            record = data["ip_reputation"].get(ip_address)
            if record is None:
                return json.dumps({"error": f"No reputation data for IP {ip_address}"})
            return json.dumps(record)

        @registry.tool(
            "Look up reputation, registration, DNS, and SSL details for a domain"
        )
        def lookup_domain(domain: str) -> str:
            record = data["domains"].get(domain)
            if record is None:
                return json.dumps({"error": f"No reputation data for domain {domain}"})
            return json.dumps(record)

        @registry.tool(
            "Look up malware/file reputation for a file hash (MD5 or SHA256)"
        )
        def lookup_file_hash(file_hash: str) -> str:
            record = data["file_hashes"].get(file_hash)
            if record is None:
                return json.dumps({"error": f"No reputation data for hash {file_hash}"})
            return json.dumps(record)

        @registry.tool("Look up CVE details by CVE ID (e.g. CVE-2024-3094)")
        def lookup_cve(cve_id: str) -> str:
            record = data["cves"].get(cve_id)
            if record is None:
                return json.dumps({"error": f"No CVE record for {cve_id}"})
            return json.dumps(record)

        @registry.tool(
            "Search MITRE ATT&CK techniques by technique ID or keyword in "
            "name/tactic/description"
        )
        def get_mitre_technique(query: str) -> str:
            query_lower = query.lower()
            matches = [
                t
                for tid, t in data["mitre_techniques"].items()
                if query_lower == tid.lower()
                or query_lower in t["name"].lower()
                or query_lower in t["tactic"].lower()
                or query_lower in t["description"].lower()
            ]
            return json.dumps({"query": query, "techniques": matches})

        @registry.tool("List all currently blocked indicators")
        def list_blocklist() -> str:
            return json.dumps({"blocked": list(data["blocklist"].values())})

        @registry.tool("Get a single security incident by ID")
        def get_incident(incident_id: str) -> str:
            record = data["incidents"].get(incident_id)
            if record is None:
                return json.dumps({"error": f"Incident {incident_id} not found"})
            return json.dumps(record)

        @registry.tool("List all security incidents")
        def list_incidents() -> str:
            return json.dumps({"incidents": list(data["incidents"].values())})

        @registry.tool(
            "Block a network indicator (ip or domain) so it is denied at the perimeter"
        )
        def block_indicator(
            indicator: str, indicator_type: str, reason: str = ""
        ) -> str:
            if indicator_type not in ("ip", "domain"):
                return json.dumps(
                    {
                        "error": "indicator_type must be 'ip' or 'domain'",
                        "status_code": 422,
                    }
                )
            if indicator in data["allowlist"]:
                return json.dumps(
                    {
                        "error": (
                            f"{indicator} is on the internal allowlist "
                            f"({data['allowlist'][indicator]['reason']}) and "
                            "cannot be blocked"
                        ),
                        "status_code": 403,
                    }
                )
            if indicator in data["blocklist"]:
                return json.dumps(
                    {
                        "error": f"{indicator} is already blocked",
                        "status_code": 409,
                    }
                )
            data["blocklist"][indicator] = {
                "indicator": indicator,
                "indicator_type": indicator_type,
                "reason": reason,
                "blocked_at": "2026-09-11",
            }
            return json.dumps(
                {
                    "status": "success",
                    "indicator": indicator,
                    "indicator_type": indicator_type,
                    "blocked": True,
                }
            )

        @registry.tool("Remove an indicator from the blocklist")
        def unblock_indicator(indicator: str) -> str:
            if indicator not in data["blocklist"]:
                return json.dumps({"error": f"{indicator} is not currently blocked"})
            del data["blocklist"][indicator]
            return json.dumps(
                {"status": "success", "indicator": indicator, "blocked": False}
            )

        @registry.tool("Open a new security incident for an indicator")
        def open_incident(title: str, indicator: str, severity: str = "medium") -> str:
            if severity not in _VALID_SEVERITIES:
                return json.dumps(
                    {
                        "error": (
                            f"Invalid severity {severity!r}; must be one of "
                            f"{list(_VALID_SEVERITIES)}"
                        ),
                        "status_code": 422,
                    }
                )
            incident_id = _incident_id()
            while incident_id in data["incidents"]:
                incident_id = _incident_id()
            indicator_type = "ip" if indicator in data["ip_reputation"] else "domain"
            data["incidents"][incident_id] = {
                "incident_id": incident_id,
                "title": title,
                "indicator": indicator,
                "indicator_type": indicator_type,
                "severity": severity,
                "status": "Open",
                "notes": [],
                "created_at": "2026-09-11",
                "triaged_at": None,
                "closed_at": None,
                "resolution": None,
            }
            return json.dumps(data["incidents"][incident_id])

        @registry.tool("Mark an open incident as triaged, with investigation notes")
        def triage_incident(incident_id: str, notes: str = "") -> str:
            incident = data["incidents"].get(incident_id)
            if incident is None:
                return json.dumps({"error": f"Incident {incident_id} not found"})
            if incident["status"] != "Open":
                return json.dumps(
                    {
                        "error": (
                            f"Incident {incident_id} is not Open "
                            f"(current status: {incident['status']}); only Open "
                            "incidents can be triaged"
                        ),
                        "status_code": 409,
                    }
                )
            incident["status"] = "Triaged"
            incident["triaged_at"] = "2026-09-11"
            if notes:
                incident["notes"].append(notes)
            return json.dumps(incident)

        @registry.tool("Close a triaged incident with a resolution summary")
        def close_incident(incident_id: str, resolution: str) -> str:
            incident = data["incidents"].get(incident_id)
            if incident is None:
                return json.dumps({"error": f"Incident {incident_id} not found"})
            if incident["status"] == "Closed":
                return json.dumps(
                    {
                        "error": f"Incident {incident_id} is already closed",
                        "status_code": 409,
                    }
                )
            if incident["status"] != "Triaged":
                return json.dumps(
                    {
                        "error": (
                            f"Incident {incident_id} must be triaged before it "
                            f"can be closed (current status: {incident['status']})"
                        ),
                        "status_code": 422,
                    }
                )
            incident["status"] = "Closed"
            incident["closed_at"] = "2026-09-11"
            incident["resolution"] = resolution
            return json.dumps(incident)
