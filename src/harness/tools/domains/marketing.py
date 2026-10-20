"""Marketing Agent stateful tool domain.

Provides tools simulating a marketing campaign analytics workflow:
campaign performance metrics (ROI, CTR, impressions), customer engagement
metrics (most/least engaged, top opens), quarterly expenses, and campaign
creation targeting engaged/disengaged users.

Data modeled after real Salesforce Marketing Agent with 5 marketing users,
30 campaigns (6 per user), and engagement metrics per campaign.
"""

from __future__ import annotations

import json

from harness.tools.environment import ToolEnvironment, register_domain
from harness.tools.registry import ToolRegistry


@register_domain("marketing")
class MarketingEnvironment(ToolEnvironment):
    def default_state(self) -> dict:
        return {
            "users": {
                "emily.carter@example.com": {
                    "name": "Emily Carter",
                    "owner_id": "005-EC",
                },
                "james.patel@example.com": {
                    "name": "James Patel",
                    "owner_id": "005-JP",
                },
                "olivia.nguyen@example.com": {
                    "name": "Olivia Nguyen",
                    "owner_id": "005-ON",
                },
                "marcus.lee@example.com": {"name": "Marcus Lee", "owner_id": "005-ML"},
                "sophia.thompson@example.com": {
                    "name": "Sophia Thompson",
                    "owner_id": "005-ST",
                },
            },
            "campaigns": {
                "MA-001": {
                    "id": "MA-001",
                    "name": "Spring Savings Blast",
                    "owner": "Emily Carter",
                    "owner_email": "emily.carter@example.com",
                    "active": True,
                    "channel": "Email",
                    "start_date": "2025-03-15",
                    "end_date": "2025-04-15",
                    "total_ad_spend": 32037.58,
                    "revenue_generated": 7013.71,
                },
                "MA-002": {
                    "id": "MA-002",
                    "name": "VIP Early Access Invite",
                    "owner": "Emily Carter",
                    "owner_email": "emily.carter@example.com",
                    "active": False,
                    "channel": "Email",
                    "start_date": "2025-04-01",
                    "end_date": "2025-05-30",
                    "total_ad_spend": 27376.23,
                    "revenue_generated": 69837.83,
                },
                "MA-003": {
                    "id": "MA-003",
                    "name": "TravelTuesday Giveaways",
                    "owner": "Emily Carter",
                    "owner_email": "emily.carter@example.com",
                    "active": True,
                    "channel": "Social Media",
                    "start_date": "2025-04-02",
                    "end_date": "2025-05-28",
                    "total_ad_spend": 16740.94,
                    "revenue_generated": 31173.23,
                },
                "MA-004": {
                    "id": "MA-004",
                    "name": "Influencer Collab Launch",
                    "owner": "Emily Carter",
                    "owner_email": "emily.carter@example.com",
                    "active": False,
                    "channel": "Social Media",
                    "start_date": "2025-02-01",
                    "end_date": "2025-05-15",
                    "total_ad_spend": 19176.62,
                    "revenue_generated": 51044.41,
                },
                "MA-005": {
                    "id": "MA-005",
                    "name": "Welcome Back Postcard Promo",
                    "owner": "Emily Carter",
                    "owner_email": "emily.carter@example.com",
                    "active": True,
                    "channel": "Direct",
                    "start_date": "2025-04-01",
                    "end_date": "2025-12-20",
                    "total_ad_spend": 12147.57,
                    "revenue_generated": 62728.12,
                },
                "MA-006": {
                    "id": "MA-006",
                    "name": "Thank You Loyalty Card",
                    "owner": "Emily Carter",
                    "owner_email": "emily.carter@example.com",
                    "active": False,
                    "channel": "Direct",
                    "start_date": "2025-03-15",
                    "end_date": "2025-04-15",
                    "total_ad_spend": 2885.57,
                    "revenue_generated": 23001.03,
                },
                "MA-007": {
                    "id": "MA-007",
                    "name": "Spring Savings Blast",
                    "owner": "James Patel",
                    "owner_email": "james.patel@example.com",
                    "active": True,
                    "channel": "Email",
                    "start_date": "2025-04-01",
                    "end_date": "2025-05-30",
                    "total_ad_spend": 7750.00,
                    "revenue_generated": 38069.74,
                },
                "MA-008": {
                    "id": "MA-008",
                    "name": "VIP Early Access Invite",
                    "owner": "James Patel",
                    "owner_email": "james.patel@example.com",
                    "active": False,
                    "channel": "Email",
                    "start_date": "2025-04-02",
                    "end_date": "2025-05-28",
                    "total_ad_spend": 12298.51,
                    "revenue_generated": 65484.00,
                },
                "MA-009": {
                    "id": "MA-009",
                    "name": "TravelTuesday Giveaways",
                    "owner": "James Patel",
                    "owner_email": "james.patel@example.com",
                    "active": True,
                    "channel": "Social Media",
                    "start_date": "2025-02-01",
                    "end_date": "2025-05-15",
                    "total_ad_spend": 26187.39,
                    "revenue_generated": 10040.38,
                },
                "MA-010": {
                    "id": "MA-010",
                    "name": "Influencer Collab Launch",
                    "owner": "James Patel",
                    "owner_email": "james.patel@example.com",
                    "active": False,
                    "channel": "Social Media",
                    "start_date": "2025-04-01",
                    "end_date": "2025-12-20",
                    "total_ad_spend": 27603.60,
                    "revenue_generated": 37916.15,
                },
                "MA-011": {
                    "id": "MA-011",
                    "name": "Welcome Back Postcard Promo",
                    "owner": "James Patel",
                    "owner_email": "james.patel@example.com",
                    "active": True,
                    "channel": "Direct",
                    "start_date": "2025-03-15",
                    "end_date": "2025-04-15",
                    "total_ad_spend": 37554.45,
                    "revenue_generated": 16290.83,
                },
                "MA-012": {
                    "id": "MA-012",
                    "name": "Thank You Loyalty Card",
                    "owner": "James Patel",
                    "owner_email": "james.patel@example.com",
                    "active": False,
                    "channel": "Direct",
                    "start_date": "2025-04-01",
                    "end_date": "2025-05-30",
                    "total_ad_spend": 6223.72,
                    "revenue_generated": 46657.51,
                },
                "MA-013": {
                    "id": "MA-013",
                    "name": "Spring Savings Blast",
                    "owner": "Olivia Nguyen",
                    "owner_email": "olivia.nguyen@example.com",
                    "active": True,
                    "channel": "Email",
                    "start_date": "2025-04-02",
                    "end_date": "2025-05-28",
                    "total_ad_spend": 1776.92,
                    "revenue_generated": 80324.25,
                },
                "MA-014": {
                    "id": "MA-014",
                    "name": "VIP Early Access Invite",
                    "owner": "Olivia Nguyen",
                    "owner_email": "olivia.nguyen@example.com",
                    "active": False,
                    "channel": "Email",
                    "start_date": "2025-02-01",
                    "end_date": "2025-05-15",
                    "total_ad_spend": 30093.57,
                    "revenue_generated": 66169.30,
                },
                "MA-015": {
                    "id": "MA-015",
                    "name": "TravelTuesday Giveaways",
                    "owner": "Olivia Nguyen",
                    "owner_email": "olivia.nguyen@example.com",
                    "active": True,
                    "channel": "Social Media",
                    "start_date": "2025-04-01",
                    "end_date": "2025-12-20",
                    "total_ad_spend": 8554.07,
                    "revenue_generated": 99520.72,
                },
                "MA-016": {
                    "id": "MA-016",
                    "name": "Influencer Collab Launch",
                    "owner": "Olivia Nguyen",
                    "owner_email": "olivia.nguyen@example.com",
                    "active": False,
                    "channel": "Social Media",
                    "start_date": "2025-03-15",
                    "end_date": "2025-04-15",
                    "total_ad_spend": 40333.97,
                    "revenue_generated": 95161.84,
                },
                "MA-017": {
                    "id": "MA-017",
                    "name": "Welcome Back Postcard Promo",
                    "owner": "Olivia Nguyen",
                    "owner_email": "olivia.nguyen@example.com",
                    "active": True,
                    "channel": "Direct",
                    "start_date": "2025-04-01",
                    "end_date": "2025-05-30",
                    "total_ad_spend": 48791.59,
                    "revenue_generated": 95822.42,
                },
                "MA-018": {
                    "id": "MA-018",
                    "name": "Thank You Loyalty Card",
                    "owner": "Olivia Nguyen",
                    "owner_email": "olivia.nguyen@example.com",
                    "active": False,
                    "channel": "Direct",
                    "start_date": "2025-04-02",
                    "end_date": "2025-05-28",
                    "total_ad_spend": 22595.65,
                    "revenue_generated": 66183.47,
                },
                "MA-019": {
                    "id": "MA-019",
                    "name": "Spring Savings Blast",
                    "owner": "Marcus Lee",
                    "owner_email": "marcus.lee@example.com",
                    "active": True,
                    "channel": "Email",
                    "start_date": "2025-02-01",
                    "end_date": "2025-05-15",
                    "total_ad_spend": 3925.67,
                    "revenue_generated": 79647.22,
                },
                "MA-020": {
                    "id": "MA-020",
                    "name": "VIP Early Access Invite",
                    "owner": "Marcus Lee",
                    "owner_email": "marcus.lee@example.com",
                    "active": False,
                    "channel": "Email",
                    "start_date": "2025-04-01",
                    "end_date": "2025-12-20",
                    "total_ad_spend": 20831.58,
                    "revenue_generated": 26677.30,
                },
                "MA-021": {
                    "id": "MA-021",
                    "name": "TravelTuesday Giveaways",
                    "owner": "Marcus Lee",
                    "owner_email": "marcus.lee@example.com",
                    "active": True,
                    "channel": "Social Media",
                    "start_date": "2025-03-15",
                    "end_date": "2025-04-15",
                    "total_ad_spend": 25743.84,
                    "revenue_generated": 17512.35,
                },
                "MA-022": {
                    "id": "MA-022",
                    "name": "Influencer Collab Launch",
                    "owner": "Marcus Lee",
                    "owner_email": "marcus.lee@example.com",
                    "active": False,
                    "channel": "Social Media",
                    "start_date": "2025-04-01",
                    "end_date": "2025-05-30",
                    "total_ad_spend": 31235.71,
                    "revenue_generated": 68437.40,
                },
                "MA-023": {
                    "id": "MA-023",
                    "name": "Welcome Back Postcard Promo",
                    "owner": "Marcus Lee",
                    "owner_email": "marcus.lee@example.com",
                    "active": True,
                    "channel": "Direct",
                    "start_date": "2025-04-02",
                    "end_date": "2025-05-28",
                    "total_ad_spend": 14562.45,
                    "revenue_generated": 35276.68,
                },
                "MA-024": {
                    "id": "MA-024",
                    "name": "Thank You Loyalty Card",
                    "owner": "Marcus Lee",
                    "owner_email": "marcus.lee@example.com",
                    "active": False,
                    "channel": "Direct",
                    "start_date": "2025-02-01",
                    "end_date": "2025-05-15",
                    "total_ad_spend": 16702.11,
                    "revenue_generated": 81830.85,
                },
                "MA-025": {
                    "id": "MA-025",
                    "name": "Spring Savings Blast",
                    "owner": "Sophia Thompson",
                    "owner_email": "sophia.thompson@example.com",
                    "active": True,
                    "channel": "Email",
                    "start_date": "2025-04-01",
                    "end_date": "2025-12-20",
                    "total_ad_spend": 36079.13,
                    "revenue_generated": 56393.30,
                },
                "MA-026": {
                    "id": "MA-026",
                    "name": "VIP Early Access Invite",
                    "owner": "Sophia Thompson",
                    "owner_email": "sophia.thompson@example.com",
                    "active": False,
                    "channel": "Email",
                    "start_date": "2025-03-15",
                    "end_date": "2025-04-15",
                    "total_ad_spend": 27918.92,
                    "revenue_generated": 86554.24,
                },
                "MA-027": {
                    "id": "MA-027",
                    "name": "TravelTuesday Giveaways",
                    "owner": "Sophia Thompson",
                    "owner_email": "sophia.thompson@example.com",
                    "active": True,
                    "channel": "Social Media",
                    "start_date": "2025-04-01",
                    "end_date": "2025-05-30",
                    "total_ad_spend": 29633.50,
                    "revenue_generated": 8957.96,
                },
                "MA-028": {
                    "id": "MA-028",
                    "name": "Influencer Collab Launch",
                    "owner": "Sophia Thompson",
                    "owner_email": "sophia.thompson@example.com",
                    "active": False,
                    "channel": "Social Media",
                    "start_date": "2025-04-02",
                    "end_date": "2025-05-28",
                    "total_ad_spend": 36195.04,
                    "revenue_generated": 65130.46,
                },
                "MA-029": {
                    "id": "MA-029",
                    "name": "Welcome Back Postcard Promo",
                    "owner": "Sophia Thompson",
                    "owner_email": "sophia.thompson@example.com",
                    "active": True,
                    "channel": "Direct",
                    "start_date": "2025-02-01",
                    "end_date": "2025-05-15",
                    "total_ad_spend": 8895.84,
                    "revenue_generated": 32912.93,
                },
                "MA-030": {
                    "id": "MA-030",
                    "name": "Thank You Loyalty Card",
                    "owner": "Sophia Thompson",
                    "owner_email": "sophia.thompson@example.com",
                    "active": False,
                    "channel": "Direct",
                    "start_date": "2025-04-01",
                    "end_date": "2025-12-20",
                    "total_ad_spend": 22642.18,
                    "revenue_generated": 2453.59,
                },
            },
            "engagement": {
                "MA-001": [
                    # Impressions must be >= clicks per row (you can't be clicked more
                    # times than shown). Prior data had clicks > impressions on several
                    # rows, producing an impossible aggregate CTR of 159.7%. Impressions
                    # raised to keep each row valid; aggregate CTR is now 88.4%
                    # (214 clicks / 242 impressions) — still Emily's best-CTR campaign
                    # and still below MA-002's total impressions. opens/clicks/conversions
                    # are unchanged, so engagement ordering and ROI are unaffected.
                    {
                        "email": "candice.johnson@example.com",
                        "sends": 4,
                        "opens": 2,
                        "clicks": 2,
                        "bounces": 1,
                        "impressions": 3,
                        "conversions": 4,
                    },
                    {
                        "email": "jason.lopez@example.com",
                        "sends": 6,
                        "opens": 5,
                        "clicks": 5,
                        "bounces": 1,
                        "impressions": 6,
                        "conversions": 3,
                    },
                    {
                        "email": "dana.wilson@example.com",
                        "sends": 10,
                        "opens": 7,
                        "clicks": 7,
                        "bounces": 3,
                        "impressions": 8,
                        "conversions": 4,
                    },
                    {
                        "email": "hannah.mitchell@example.com",
                        "sends": 12,
                        "opens": 8,
                        "clicks": 8,
                        "bounces": 3,
                        "impressions": 9,
                        "conversions": 3,
                    },
                    {
                        "email": "james.white@example.com",
                        "sends": 63,
                        "opens": 29,
                        "clicks": 14,
                        "bounces": 10,
                        "impressions": 16,
                        "conversions": 10,
                    },
                    {
                        "email": "brian.turner@example.com",
                        "sends": 66,
                        "opens": 47,
                        "clicks": 30,
                        "bounces": 20,
                        "impressions": 34,
                        "conversions": 5,
                    },
                    {
                        "email": "kelly.bridges@example.com",
                        "sends": 77,
                        "opens": 56,
                        "clicks": 35,
                        "bounces": 10,
                        "impressions": 39,
                        "conversions": 5,
                    },
                    {
                        "email": "jimmy.carter@example.com",
                        "sends": 99,
                        "opens": 85,
                        "clicks": 49,
                        "bounces": 20,
                        "impressions": 55,
                        "conversions": 5,
                    },
                    {
                        "email": "anne.leach@example.com",
                        "sends": 93,
                        "opens": 83,
                        "clicks": 38,
                        "bounces": 30,
                        "impressions": 43,
                        "conversions": 20,
                    },
                    {
                        "email": "teresa.flowers@example.com",
                        "sends": 83,
                        "opens": 57,
                        "clicks": 26,
                        "bounces": 30,
                        "impressions": 29,
                        "conversions": 12,
                    },
                ],
                "MA-002": [
                    {
                        "email": "candice.johnson@example.com",
                        "sends": 50,
                        "opens": 40,
                        "clicks": 25,
                        "bounces": 5,
                        "impressions": 30,
                        "conversions": 15,
                    },
                    {
                        "email": "jason.lopez@example.com",
                        "sends": 45,
                        "opens": 35,
                        "clicks": 20,
                        "bounces": 3,
                        "impressions": 25,
                        "conversions": 12,
                    },
                    {
                        "email": "dana.wilson@example.com",
                        "sends": 60,
                        "opens": 50,
                        "clicks": 35,
                        "bounces": 8,
                        "impressions": 40,
                        "conversions": 20,
                    },
                    {
                        "email": "hannah.mitchell@example.com",
                        "sends": 55,
                        "opens": 42,
                        "clicks": 28,
                        "bounces": 6,
                        "impressions": 35,
                        "conversions": 18,
                    },
                    {
                        "email": "james.white@example.com",
                        "sends": 70,
                        "opens": 60,
                        "clicks": 40,
                        "bounces": 10,
                        "impressions": 50,
                        "conversions": 25,
                    },
                    {
                        "email": "brian.turner@example.com",
                        "sends": 40,
                        "opens": 30,
                        "clicks": 18,
                        "bounces": 4,
                        "impressions": 20,
                        "conversions": 10,
                    },
                    {
                        "email": "kelly.bridges@example.com",
                        "sends": 80,
                        "opens": 70,
                        "clicks": 50,
                        "bounces": 12,
                        "impressions": 60,
                        "conversions": 30,
                    },
                    {
                        "email": "jimmy.carter@example.com",
                        "sends": 35,
                        "opens": 25,
                        "clicks": 15,
                        "bounces": 2,
                        "impressions": 18,
                        "conversions": 8,
                    },
                    {
                        "email": "anne.leach@example.com",
                        "sends": 65,
                        "opens": 55,
                        "clicks": 38,
                        "bounces": 9,
                        "impressions": 45,
                        "conversions": 22,
                    },
                    {
                        "email": "teresa.flowers@example.com",
                        "sends": 48,
                        "opens": 38,
                        "clicks": 22,
                        "bounces": 5,
                        "impressions": 28,
                        "conversions": 14,
                    },
                ],
                "MA-007": [
                    {
                        "email": "michael.ross@example.com",
                        "sends": 30,
                        "opens": 20,
                        "clicks": 12,
                        "bounces": 3,
                        "impressions": 15,
                        "conversions": 8,
                    },
                    {
                        "email": "sarah.chen@example.com",
                        "sends": 55,
                        "opens": 45,
                        "clicks": 30,
                        "bounces": 5,
                        "impressions": 35,
                        "conversions": 18,
                    },
                    {
                        "email": "david.kim@example.com",
                        "sends": 42,
                        "opens": 32,
                        "clicks": 22,
                        "bounces": 4,
                        "impressions": 25,
                        "conversions": 12,
                    },
                    {
                        "email": "lisa.wang@example.com",
                        "sends": 68,
                        "opens": 58,
                        "clicks": 40,
                        "bounces": 8,
                        "impressions": 45,
                        "conversions": 22,
                    },
                    {
                        "email": "robert.jones@example.com",
                        "sends": 25,
                        "opens": 15,
                        "clicks": 8,
                        "bounces": 2,
                        "impressions": 10,
                        "conversions": 5,
                    },
                ],
                "MA-013": [
                    {
                        "email": "emma.davis@example.com",
                        "sends": 75,
                        "opens": 65,
                        "clicks": 45,
                        "bounces": 8,
                        "impressions": 55,
                        "conversions": 28,
                    },
                    {
                        "email": "noah.wilson@example.com",
                        "sends": 40,
                        "opens": 30,
                        "clicks": 18,
                        "bounces": 4,
                        "impressions": 22,
                        "conversions": 10,
                    },
                    {
                        "email": "olivia.brown@example.com",
                        "sends": 90,
                        "opens": 80,
                        "clicks": 55,
                        "bounces": 10,
                        "impressions": 70,
                        "conversions": 35,
                    },
                    {
                        "email": "liam.garcia@example.com",
                        "sends": 55,
                        "opens": 42,
                        "clicks": 28,
                        "bounces": 6,
                        "impressions": 35,
                        "conversions": 16,
                    },
                    {
                        "email": "sophia.martinez@example.com",
                        "sends": 62,
                        "opens": 50,
                        "clicks": 33,
                        "bounces": 7,
                        "impressions": 40,
                        "conversions": 20,
                    },
                ],
                "MA-019": [
                    {
                        "email": "alex.thompson@example.com",
                        "sends": 48,
                        "opens": 38,
                        "clicks": 25,
                        "bounces": 5,
                        "impressions": 30,
                        "conversions": 15,
                    },
                    {
                        "email": "mia.johnson@example.com",
                        "sends": 72,
                        "opens": 62,
                        "clicks": 42,
                        "bounces": 8,
                        "impressions": 50,
                        "conversions": 25,
                    },
                    {
                        "email": "ethan.williams@example.com",
                        "sends": 35,
                        "opens": 25,
                        "clicks": 15,
                        "bounces": 3,
                        "impressions": 18,
                        "conversions": 8,
                    },
                    {
                        "email": "isabella.lee@example.com",
                        "sends": 60,
                        "opens": 50,
                        "clicks": 35,
                        "bounces": 6,
                        "impressions": 40,
                        "conversions": 20,
                    },
                    {
                        "email": "james.anderson@example.com",
                        "sends": 85,
                        "opens": 75,
                        "clicks": 50,
                        "bounces": 10,
                        "impressions": 60,
                        "conversions": 30,
                    },
                ],
                "MA-025": [
                    {
                        "email": "charlotte.harris@example.com",
                        "sends": 55,
                        "opens": 45,
                        "clicks": 30,
                        "bounces": 5,
                        "impressions": 35,
                        "conversions": 18,
                    },
                    {
                        "email": "benjamin.clark@example.com",
                        "sends": 40,
                        "opens": 30,
                        "clicks": 18,
                        "bounces": 4,
                        "impressions": 22,
                        "conversions": 10,
                    },
                    {
                        "email": "amelia.lewis@example.com",
                        "sends": 78,
                        "opens": 68,
                        "clicks": 48,
                        "bounces": 9,
                        "impressions": 55,
                        "conversions": 28,
                    },
                    {
                        "email": "henry.walker@example.com",
                        "sends": 32,
                        "opens": 22,
                        "clicks": 12,
                        "bounces": 3,
                        "impressions": 15,
                        "conversions": 6,
                    },
                    {
                        "email": "evelyn.hall@example.com",
                        "sends": 65,
                        "opens": 55,
                        "clicks": 38,
                        "bounces": 7,
                        "impressions": 42,
                        "conversions": 22,
                    },
                ],
            },
            "created_campaigns": [],
        }

    def _get_user_campaigns(self, email: str) -> list[dict]:
        return [
            c
            for c in self.data["campaigns"].values()
            if c["owner_email"].lower() == email.lower()
        ]

    def _validate_user(self, email: str) -> str | None:
        if email.lower() not in {k.lower() for k in self.data["users"]}:
            return f"User with email {email} not found"
        return None

    def register_tools(self, registry: ToolRegistry) -> None:
        env = self

        @registry.tool("Get the ROI of a campaign based on campaign ID")
        def MA_Get_ROI_A_Of_Campaign(campaignId: str, email: str) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            campaign = env.data["campaigns"].get(campaignId)
            if not campaign:
                return json.dumps({"error": f"Campaign {campaignId} not found"})
            if campaign["owner_email"].lower() != email.lower():
                return json.dumps(
                    {
                        "error": f"Campaign {campaignId} does not belong to user {email}",
                        "status_code": 403,
                    }
                )
            spend = campaign["total_ad_spend"]
            revenue = campaign["revenue_generated"]
            roi = ((revenue - spend) / spend) * 100 if spend > 0 else 0
            return json.dumps(
                {
                    "output": (
                        f"Campaign '{campaign['name']}' (ID: {campaignId}) — "
                        f"Total Ad Spend: ${spend:,.2f}, Revenue Generated: ${revenue:,.2f}, "
                        f"ROI: {roi:.1f}%"
                    )
                }
            )

        @registry.tool(
            "Get the campaign with the highest impressions for a marketing user"
        )
        def MA_Get_Campaign_With_High_Impression(email: str) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            user_campaigns = env._get_user_campaigns(email)
            if not user_campaigns:
                return json.dumps({"error": f"No campaigns found for {email}"})
            best = None
            best_impressions = -1
            for c in user_campaigns:
                eng = env.data["engagement"].get(c["id"], [])
                total_imp = sum(e["impressions"] for e in eng)
                if total_imp > best_impressions:
                    best_impressions = total_imp
                    best = c
            assert best is not None  # user_campaigns is non-empty ⇒ loop assigns
            return json.dumps(
                {
                    "output": (
                        f"Campaign with highest impressions for {email}: "
                        f"'{best['name']}' (ID: {best['id']}) with {best_impressions} total impressions"
                    )
                }
            )

        @registry.tool(
            "Get the campaign with the least impressions for a marketing user"
        )
        def MA_Get_Campaign_With_Least_Impression(email: str) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            user_campaigns = env._get_user_campaigns(email)
            if not user_campaigns:
                return json.dumps({"error": f"No campaigns found for {email}"})
            worst = None
            worst_impressions = float("inf")
            for c in user_campaigns:
                eng = env.data["engagement"].get(c["id"], [])
                total_imp = sum(e["impressions"] for e in eng)
                if total_imp < worst_impressions:
                    worst_impressions = total_imp
                    worst = c
            assert worst is not None  # user_campaigns is non-empty ⇒ loop assigns
            return json.dumps(
                {
                    "output": (
                        f"Campaign with least impressions for {email}: "
                        f"'{worst['name']}' (ID: {worst['id']}) with {int(worst_impressions)} total impressions"
                    )
                }
            )

        @registry.tool("Get the click-through rate (CTR) of a campaign")
        def MA_Get_CTR_Of_A_Campaign(campaignId: str, email: str) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            campaign = env.data["campaigns"].get(campaignId)
            if not campaign:
                return json.dumps({"error": f"Campaign {campaignId} not found"})
            if campaign["owner_email"].lower() != email.lower():
                return json.dumps(
                    {
                        "error": f"Campaign {campaignId} does not belong to user {email}",
                        "status_code": 403,
                    }
                )
            eng = env.data["engagement"].get(campaignId, [])
            total_clicks = sum(e["clicks"] for e in eng)
            total_impressions = sum(e["impressions"] for e in eng)
            ctr = (
                (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
            )
            return json.dumps(
                {
                    "output": (
                        f"Campaign '{campaign['name']}' (ID: {campaignId}) — "
                        f"Total Clicks: {total_clicks}, Total Impressions: {total_impressions}, "
                        f"CTR: {ctr:.1f}%"
                    )
                }
            )

        @registry.tool("Get the campaign with the best CTR for a marketing user")
        def MA_Get_Campaign_With_Best_CTR(email: str) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            user_campaigns = env._get_user_campaigns(email)
            if not user_campaigns:
                return json.dumps({"error": f"No campaigns found for {email}"})
            best = None
            best_ctr = -1.0
            for c in user_campaigns:
                eng = env.data["engagement"].get(c["id"], [])
                total_clicks = sum(e["clicks"] for e in eng)
                total_impressions = sum(e["impressions"] for e in eng)
                ctr = (
                    (total_clicks / total_impressions) * 100
                    if total_impressions > 0
                    else 0
                )
                if ctr > best_ctr:
                    best_ctr = ctr
                    best = c
            assert best is not None  # user_campaigns is non-empty ⇒ loop assigns
            return json.dumps(
                {
                    "output": (
                        f"Campaign with best CTR for {email}: "
                        f"'{best['name']}' (ID: {best['id']}) with CTR: {best_ctr:.1f}%"
                    )
                }
            )

        @registry.tool("Get the campaign with the worst CTR for a marketing user")
        def MA_Get_Campaign_With_Worst_CTR(email: str) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            user_campaigns = env._get_user_campaigns(email)
            if not user_campaigns:
                return json.dumps({"error": f"No campaigns found for {email}"})
            worst = None
            worst_ctr = float("inf")
            for c in user_campaigns:
                eng = env.data["engagement"].get(c["id"], [])
                total_clicks = sum(e["clicks"] for e in eng)
                total_impressions = sum(e["impressions"] for e in eng)
                ctr = (
                    (total_clicks / total_impressions) * 100
                    if total_impressions > 0
                    else 0
                )
                if ctr < worst_ctr:
                    worst_ctr = ctr
                    worst = c
            assert worst is not None  # user_campaigns is non-empty ⇒ loop assigns
            return json.dumps(
                {
                    "output": (
                        f"Campaign with worst CTR for {email}: "
                        f"'{worst['name']}' (ID: {worst['id']}) with CTR: {worst_ctr:.1f}%"
                    )
                }
            )

        @registry.tool("Get the total marketing expense for a quarter")
        def MA_Get_Total_Expense_Quarter(Email: str) -> str:
            err = env._validate_user(Email)
            if err:
                return json.dumps({"error": err})
            user_campaigns = env._get_user_campaigns(Email)
            if not user_campaigns:
                return json.dumps({"error": f"No campaigns found for {Email}"})
            q1 = sum(
                c["total_ad_spend"]
                for c in user_campaigns
                if c["start_date"] < "2025-04-01"
            )
            q2 = sum(
                c["total_ad_spend"]
                for c in user_campaigns
                if "2025-04-01" <= c["start_date"] < "2025-07-01"
            )
            total = sum(c["total_ad_spend"] for c in user_campaigns)
            return json.dumps(
                {
                    "outPut": (
                        f"Marketing expenses for {Email}: "
                        f"Q1 2025 (Jan-Mar): ${q1:,.2f}, "
                        f"Q2 2025 (Apr-Jun): ${q2:,.2f}, "
                        f"Total: ${total:,.2f}"
                    )
                }
            )

        @registry.tool("Get the list of most or least engaged customers for a campaign")
        def MA_Get_Most_or_Least_Engaged_Customer_List(
            Type: str, campaignId: str, email: str
        ) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            campaign = env.data["campaigns"].get(campaignId)
            if not campaign:
                return json.dumps({"error": f"Campaign {campaignId} not found"})
            if campaign["owner_email"].lower() != email.lower():
                return json.dumps(
                    {
                        "error": f"Campaign {campaignId} does not belong to user {email}",
                        "status_code": 403,
                    }
                )
            eng = env.data["engagement"].get(campaignId, [])
            if not eng:
                return json.dumps(
                    {"error": f"No engagement data for campaign {campaignId}"}
                )

            def engagement_score(e: dict) -> int:
                return e["opens"] + e["clicks"] + e["conversions"]

            sorted_eng = sorted(
                eng, key=engagement_score, reverse=(Type.lower() == "most")
            )
            top_5 = sorted_eng[:5]
            customer_list = [
                {
                    "email": e["email"],
                    "opens": e["opens"],
                    "clicks": e["clicks"],
                    "conversions": e["conversions"],
                }
                for e in top_5
            ]
            return json.dumps(
                {
                    "CustomerList": customer_list,
                    "GeneralResponse": f"Top 5 {Type.lower()} engaged customers for campaign '{campaign['name']}' (ID: {campaignId})",
                }
            )

        @registry.tool(
            "Get the list of customers with the highest number of opens for a campaign"
        )
        def MA_Get_Customer_List_Based_On_Top_Open(campaignId: str, email: str) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            campaign = env.data["campaigns"].get(campaignId)
            if not campaign:
                return json.dumps({"error": f"Campaign {campaignId} not found"})
            if campaign["owner_email"].lower() != email.lower():
                return json.dumps(
                    {
                        "error": f"Campaign {campaignId} does not belong to user {email}",
                        "status_code": 403,
                    }
                )
            eng = env.data["engagement"].get(campaignId, [])
            if not eng:
                return json.dumps(
                    {"error": f"No engagement data for campaign {campaignId}"}
                )
            sorted_eng = sorted(eng, key=lambda e: e["opens"], reverse=True)
            top_5 = sorted_eng[:5]
            customer_list = [{"email": e["email"], "opens": e["opens"]} for e in top_5]
            return json.dumps(
                {
                    "CustomerList": customer_list,
                    "output": f"Top 5 customers by opens for campaign '{campaign['name']}' (ID: {campaignId})",
                }
            )

        @registry.tool("Create a campaign targeting most or least engaged customers")
        def MA_Create_A_Campaign_With_Most_or_Least_Engaged_Customers(
            campaignId: str,
            email: str,
            inputCampaignName: str,
            inputDescription: str,
            inputStartDate: str,
            inputEndDate: str,
            Type: str = "Most",
        ) -> str:
            err = env._validate_user(email)
            if err:
                return json.dumps({"error": err})
            campaign = env.data["campaigns"].get(campaignId)
            if not campaign:
                return json.dumps({"error": f"Source campaign {campaignId} not found"})
            if campaign["owner_email"].lower() != email.lower():
                return json.dumps(
                    {
                        "error": f"Campaign {campaignId} does not belong to user {email}",
                        "status_code": 403,
                    }
                )
            eng = env.data["engagement"].get(campaignId, [])
            if not eng:
                return json.dumps(
                    {"error": f"No engagement data for campaign {campaignId}"}
                )

            def engagement_score(e: dict) -> int:
                return e["opens"] + e["clicks"] + e["conversions"]

            sorted_eng = sorted(
                eng, key=engagement_score, reverse=(Type.lower() == "most")
            )
            top_5 = sorted_eng[:5]
            customer_list = [
                {
                    "email": e["email"],
                    "opens": e["opens"],
                    "clicks": e["clicks"],
                    "conversions": e["conversions"],
                }
                for e in top_5
            ]

            new_campaign = {
                "name": inputCampaignName,
                "description": inputDescription,
                "start_date": inputStartDate,
                "end_date": inputEndDate,
                "type": Type,
                "source_campaign_id": campaignId,
                "target_customers": customer_list,
                "owner_email": email,
            }
            env.data["created_campaigns"].append(new_campaign)

            return json.dumps(
                {
                    "CustomerList": customer_list,
                    "GeneralResponse": (
                        f"Campaign '{inputCampaignName}' created successfully targeting "
                        f"{Type.lower()} engaged customers from '{campaign['name']}'. "
                        f"Start: {inputStartDate}, End: {inputEndDate}. "
                        f"{len(customer_list)} customers targeted."
                    ),
                }
            )
