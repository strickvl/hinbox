# Entity Examples

## People
```json
{
  "name": "Abd al Rahim al Nashiri",
  "type": "detainee",
  "profile": {
    "text": "Abd al Rahim al Nashiri is an alleged USS Cole bomber who was held as a captive at Guantánamo Bay^[fb46f019-6ead-40d1-a9db-0f6198d5e262]. He was interrogated in Thailand, reportedly under the oversight of Gina Haspel^[fb46f019-6ead-40d1-a9db-0f6198d5e262].",
    "tags": [
      "Abd al Rahim al Nashiri",
      "Guantanamo Bay",
      "CIA",
      "Terrorism",
      "USS Cole bombing"
    ],
    "confidence": 0.7,
    "sources": [
      "fb46f019-6ead-40d1-a9db-0f6198d5e262"
    ]
  },
  "articles": [
    {
      "article_id": "fb46f019-6ead-40d1-a9db-0f6198d5e262",
      "article_title": "Did CIA Director Gina Haspel run a black site at Guantánamo?",
      "article_url": "https://www.miamiherald.com/news/nation-world/national/article223835570.html",
      "article_published_date": 1546959319
    }
  ],
  "extraction_timestamp": "2025-03-13T15:51:24.773054"
}
```

## Events
```json
{
  "title": "Agreement to conceal torture details",
  "description": "Khalid Shaikh Mohammed agreed to never disclose secret aspects of his torture by the CIA if he is allowed to plead guilty rather than face a death-penalty trial.",
  "event_type": "legal_decision",
  "start_date": "2024-07-31T00:00:00+00:00",
  "end_date": null,
  "is_fuzzy_date": false,
  "tags": [
    "torture",
    "interrogation",
    "policy_change",
    "legal_challenge"
  ],
  "profile": {
    "text": "The 'Agreement to conceal torture details' is a legal settlement in which Khalid Shaikh Mohammed, accused of plotting the 9/11 attacks, agreed to not disclose secret aspects of his torture by the CIA in exchange for being allowed to plead guilty and avoid a death-penalty trial^[0532a14f-a9af-4665-b955-293fe6aa874f]. The agreement includes a clause preventing Mohammed from publicly identifying people, places, and other details from his time in the CIA’s secret prisons overseas from 2003-06^[0532a14f-a9af-4665-b955-293fe6aa874f].\n\nThe agreement, part of a 20-page settlement, states that Mohammed agreed not to disclose information about his “capture, detention, confinement of himself or others” while in U.S. custody^[0532a14f-a9af-4665-b955-293fe6aa874f]. He signed the agreement on July 31^[0532a14f-a9af-4665-b955-293fe6aa874f], after over two years of negotiations between his lawyers and prosecutors^[0532a14f-a9af-4665-b955-293fe6aa874f]. Former Defense Secretary Lloyd Austin III attempted to withdraw from the settlement on August 2, but two military courts ruled his action was too late^[0532a14f-a9af-4665-b955-293fe6aa874f].\n\nThe settlement, along with similar agreements with co-defendants Walid bin Attash and Mustafa al-Hawsawi, remains mostly under seal while defense lawyers seek to enforce the agreement and hold a sentencing hearing at Guantánamo Bay^[0532a14f-a9af-4665-b955-293fe6aa874f]. The agreement allows Mohammed to discuss the details with his lawyers for the sentencing trial, but they are obligated to keep the information classified^[0532a14f-a9af-4665-b955-293fe6aa874f]. A three-judge panel of the U.S. Court of Appeals for the D.C. Circuit extended an order preventing the military judge at Guantánamo from holding plea proceedings in the case^[0532a14f-a9af-4665-b955-293fe6aa874f].",
    "tags": [
      "legal agreement",
      "torture",
      "national security",
      "Guantánamo Bay",
      "CIA black sites"
    ],
    "confidence": 0.9,
    "sources": [
      "0532a14f-a9af-4665-b955-293fe6aa874f"
    ]
  },
  "articles": [
    {
      "article_id": "0532a14f-a9af-4665-b955-293fe6aa874f",
      "article_title": "9/11 plea deal includes lifetime gag order on CIA torture secrets",
      "article_url": "https://www.miamiherald.com/news/nation-world/world/article299960709.html",
      "article_published_date": 1738985719
    }
  ],
  "extraction_timestamp": "2025-03-13T15:54:51.850902"
}
```

## Locations
```json
{
  "name": "Afghanistan",
  "type": "country",
  "profile": {
    "text": "Afghanistan is a location where the CIA reportedly operated a \"black site\" prison^[0532a14f-a9af-4665-b955-293fe6aa874f]. Khalid Shaikh Mohammed, accused of plotting the 9/11 attacks, was shuttled between prisons in Afghanistan and other locations^[0532a14f-a9af-4665-b955-293fe6aa874f]. The CIA has not acknowledged Afghanistan as a former black site^[0532a14f-a9af-4665-b955-293fe6aa874f].",
    "tags": [
      "location",
      "Afghanistan",
      "CIA black site",
      "Guantánamo Bay"
    ],
    "confidence": 0.7,
    "sources": [
      "0532a14f-a9af-4665-b955-293fe6aa874f"
    ]
  },
  "articles": [
    {
      "article_id": "0532a14f-a9af-4665-b955-293fe6aa874f",
      "article_title": "9/11 plea deal includes lifetime gag order on CIA torture secrets",
      "article_url": "https://www.miamiherald.com/news/nation-world/world/article299960709.html",
      "article_published_date": 1738985719
    }
  ],
  "extraction_timestamp": "2025-03-13T15:54:51.850902"
}
```

## Organizations
```json
{
  "name": "American Civil Liberties Union",
  "type": "legal",
  "profile": {
    "text": "The American Civil Liberties Union (ACLU) is mentioned in the context of a lawsuit against the Trump administration regarding the detention of Venezuelan migrants at Guantánamo Bay, Cuba^[17fe80a6-5dc1-4e41-a7b5-eb9f56a64461]. Lee Gelernt, an ACLU *immigrant rights lawyer*, is the lead counsel in the lawsuit^[17fe80a6-5dc1-4e41-a7b5-eb9f56a64461]. The lawsuit seeks to grant migrants held at Guantánamo Bay access to legal representation to challenge their detention^[17fe80a6-5dc1-4e41-a7b5-eb9f56a64461]. The ACLU's involvement highlights concerns about the legal basis for holding individuals arrested in the U.S. at Guantánamo for immigration detention, arguing that the government cannot circumvent detainees' rights by detaining them in an overseas facility^[17fe80a6-5dc1-4e41-a7b5-eb9f56a64461].",
    "tags": [
      "American Civil Liberties Union",
      "ACLU",
      "Immigrant Rights",
      "Legal Aid",
      "Lawsuit"
    ],
    "confidence": 0.8,
    "sources": [
      "17fe80a6-5dc1-4e41-a7b5-eb9f56a64461"
    ]
  },
  "articles": [
    {
      "article_id": "17fe80a6-5dc1-4e41-a7b5-eb9f56a64461",
      "article_title": "Some Migrants Sent by Trump to Guantánamo Are Being Held by Military Guards",
      "article_url": "https://www.miamiherald.com/news/nation-world/world/article300235379.html",
      "article_published_date": 1739415011
    }
  ],
  "extraction_timestamp": "2025-03-13T15:49:59.064376"
}
```

