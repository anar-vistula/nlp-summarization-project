"""
Long sample texts in English and Turkish for model comparison.
Each sample includes: text, reference_summary, language, category.
"""

SAMPLES = {
    "en_climate": {
        "language": "en",
        "category": "Scientific/Policy",
        "title": "Climate Change and Global Response",
        "text": (
            "Climate change refers to long-term shifts in temperatures and weather patterns. "
            "These shifts may be natural, such as through variations in the solar cycle, but since the 1800s, "
            "human activities have been the main driver of climate change, primarily due to burning fossil fuels "
            "like coal, oil and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket "
            "wrapped around the Earth, trapping the sun's heat and raising temperatures. "
            "Examples of greenhouse gas emissions that are causing climate change include carbon dioxide and methane. "
            "These come from using gasoline for driving a car or coal for heating a building, for example. "
            "Clearing land and forests can also release carbon dioxide. Landfills for garbage are a major source of methane emissions. "
            "Energy, industry, transport, buildings, agriculture and land use are among the main emitters. "
            "The Earth is now about 1.1 degrees Celsius warmer than it was in the late 1800s. The last decade was the warmest on record. "
            "Many people think climate change mainly means warmer temperatures. But temperature rise is only the beginning of the story. "
            "Because the Earth is a system, where everything is connected, changes in one area can influence changes in all others. "
            "The consequences of climate change now include, among others, intense droughts, water scarcity, severe fires, "
            "rising sea levels, flooding, melting polar ice, catastrophic storms and declining biodiversity. "
            "People are experiencing climate change in diverse ways. Climate change can affect our health, ability to grow food, "
            "housing, safety and work. Some of us are already more vulnerable to climate impacts, such as people living in "
            "small island nations and other developing countries. Conditions like sea-level rise and saltwater intrusion have "
            "advanced to the point where whole communities have had to relocate, and protracted droughts are putting people at risk of famine. "
            "In the future, the number of people displaced by weather-related events is expected to rise. "
            "Every increase in global warming matters. In a series of UN reports, thousands of scientists and government reviewers "
            "agreed that limiting global temperature rise to no more than 1.5 degrees Celsius would help us avoid the worst climate "
            "impacts and maintain a livable climate. Yet based on current national climate plans, global warming is projected to reach "
            "around 3.2 degrees Celsius by the end of the century. The emissions that cause climate change come from every part of the "
            "world and affect everyone, but some countries produce much more than others. The 100 least-emitting countries generate "
            "3 per cent of total emissions. The 10 countries with the largest emissions contribute 68 per cent. Everyone must take "
            "climate action, but people and countries creating more of the problem have a greater responsibility to act first. "
            "Many climate change solutions can deliver economic benefits while improving our lives and protecting the environment. "
            "We also have global frameworks and agreements to guide progress, such as the Sustainable Development Goals, "
            "the UN Framework Convention on Climate Change and the Paris Agreement. Three broad categories of action are: "
            "cutting emissions, adapting to climate impacts and financing required adjustments. "
            "Switching energy systems from fossil fuels to renewables like solar or wind will reduce the emissions driving climate change. "
            "But we have to start right now. While a growing coalition of countries is committing to net zero emissions by 2050, "
            "about half of emissions cuts must be in place by 2030 to keep warming below 1.5 degrees. "
            "Fossil fuel production must decline by roughly 6 per cent per year between 2020 and 2030."
        ),
        "reference_summary": (
            "Climate change is primarily caused by human activities since the 1800s, especially burning fossil fuels "
            "which release greenhouse gases like CO2 and methane. The Earth has warmed 1.1°C since the late 1800s, "
            "causing droughts, floods, rising seas, melting ice, and biodiversity loss. Vulnerable communities face "
            "displacement and famine. Scientists recommend limiting warming to 1.5°C, but current plans project 3.2°C "
            "by century's end. The top 10 emitting countries produce 68% of emissions. Solutions include switching to "
            "renewable energy, adapting to impacts, and international agreements like the Paris Agreement, with "
            "emissions needing to drop 6% annually through 2030."
        ),
    },
    "en_narrative": {
        "language": "en",
        "category": "Narrative/Story",
        "title": "The School Handyman",
        "text": (
            "At the heart of an affluent neighbourhood stood a prestigious private school. "
            "The school employed a handyman named John, who helped to keep the school tidy, maintain the buildings "
            "and move things in and out of the basement. John was a quiet man who was happy to work on his own. "
            "He whistled a tune while he worked, and never noticed that the pupils were making fun of him. "
            "He tried to make friends with his colleagues, but they always had an excuse. They were just going to "
            "their class, even though it was the end of the day. Or they had to go to lunch, even when they had just had breakfast. "
            "One day, the headmistress called John into her office. John, the school is facing a budget crisis, she said. "
            "We need to reduce our costs a lot, and we need to do it quickly. John, you are fired. "
            "But why? Do I not do a good job? asked John. You do, said the headmistress. But all the other staff have "
            "told me that they do not like you. You are unusual, and you whistle too much. I can no longer afford to pay "
            "someone so unpleasant. After John left, the headmistress ate a bar of chocolate and played darts. "
            "But the school costs were still too high. Over the next few weeks, she cancelled school lunches, stopped "
            "the music lessons and closed the sports field. Finally, she went into the science laboratory and gathered "
            "all the teachers. We have to sell all of the science and computer equipment, she explained. "
            "But it is all brand new! the Head of Science exclaimed. We only just bought it all! How are we meant to teach without it? "
            "Why do you not fire the handyman instead? asked another teacher. He does not do anything useful. "
            "The other teachers agreed. They were surprised to learn that he had left weeks ago. "
            "There is no point complaining, the headmistress concluded. Once we sell the equipment, the budget crisis will be over. "
            "You will just have to teach science like they did in nineteen-hundred. Come and help me get the blackboards "
            "and chalk out of the basement. They all arrived in the room underground. In the dark, they could see some blinking lights. "
            "The headmistress switched on the main light, and everyone gasped. The basement was filled with the school's previous "
            "science and computer equipment. The headmistress had asked John to throw it away last year, because a lot of it was broken. "
            "Instead, he had kept it all here, and had been fixing it ever since. It is working! cried the Head of Science, "
            "inspecting the equipment. Headmistress, we need John back to fix the rest of it. All of the teachers nodded their heads "
            "and stared at the headmistress. She took a deep breath, got out her phone and called John. "
            "I see, she finished, putting the phone down. What did he say? the other teachers asked. "
            "He has made a lot of friends at his new school."
        ),
        "reference_summary": (
            "A school handyman named John was fired during a budget crisis because his colleagues told the headmistress "
            "they disliked him. After his departure, costs remained high and the school had to sell equipment. "
            "When teachers went to the basement, they discovered John had secretly been repairing old broken equipment "
            "instead of throwing it away. They tried to rehire him, but he had already found happiness at a new school "
            "where he made friends."
        ),
    },
    "tr_ataturk": {
        "language": "tr",
        "category": "Historical/Memoir",
        "title": "Atatürk Anıları",
        "text": (
            "Bir akşam Atatürk, bu adı almaya karar verdiğini söyledi ve düşüncemi sordu. "
            "Mustafa Kemal adiyle parlak zaferler kazandınız, ün saldınız, çürümüş bir imparatorluktan dipdiri bir "
            "cumhuriyet çıkardınız, büyük devrimler yaptınız; bu adı bırakmak doğru olmaz dedim. "
            "Atatürk yalnız şu karşılığı verdi: İbn-i Sina'ya neden kızıyorsun? Ben işi anlamış ve Doğru demiştim. "
            "Bunun anlamı şuydu: İbn-i Sina diye anılan Ebu Ali el-Hüseyin İbn-i Abdullah, Buhara yakınlarında doğmuştur. "
            "O sırada Mâveraü'n-Nehr bir Türk ülkesiydi. Atatürk bu adı almakla Türklerin de genellikle Türk adları "
            "taşımalarının gerektiğini belirtmek istemiştir. "
            "1933 yazında Atatürk'ün İstanbul'da bulunduğu sırada eski Fransız başbakanlarından Herriot oraya gelmişti. "
            "Atatürk onu Dolmabahçe Sarayında kabul etti. Herriot çok saygılıydı ve her sözünde büyük bir yakınlık duygusu seziliyordu. "
            "Türlü devlet işleri ve layiklik üzerinde konuşulurken, eski Fransız Başbakanı: Size nasıl hayran olmayayım? "
            "Halifeyi sınır dışı ettiniz; halbuki Papanın Paris temsilcisi benim hükümetimi düşürdü dedi. "
            "Atatürk, Hitler'in gerçek bir başarıya ulaşabileceğine inanmazdı ve durmadan yeni sorunlar ortaya atmasının "
            "kendisini bir gün yıkıma götürebileceğini söylerdi. Bununla birlikte, Versailles Antlaşmasının kısıtlayıcı "
            "hükümlerini ortadan kaldırmak yolundaki çabalarını doğal bulurdu. "
            "Atatürk bir devrimi gerçekleştirmeyi düşününce konuyu ve onun yaratabileceği tepkileri inceden inceye araştırır, "
            "sezdirmeden yoklamalar yapar, onu tepkisiz veya en az tepki ile halka kabul ettirmek için gereken her tedbiri alırdı. "
            "Şapka devrimi bu yolda iyi bir örnektir. Atatürk özel hayatında olduğu gibi, toplum hayatında ve siyasada da "
            "büyük bir ruhbilim ustası idi."
        ),
        "reference_summary": (
            "Bu metin Atatürk hakkında kişisel anıları içermektedir. Atatürk adını, Türklerin Türk adları taşıması "
            "gerektiğini vurgulamak için almıştır. Fransız başbakan Herriot ile Dolmabahçe'de görüşmüş ve laiklik "
            "konusunda takdir görmüştür. Hitler hakkında şüpheci bir tutum sergilemiş, onun yıkıma gideceğini "
            "öngörmüştür. Devrim stratejisinde halkın tepkisini önceden araştıran bir yaklaşım benimsemiş, "
            "şapka devrimi bunun en iyi örneğidir."
        ),
    },
    "tr_technology": {
        "language": "tr",
        "category": "Technology",
        "title": "Yapay Zeka ve Toplum",
        "text": (
            "Yapay zeka, makinelerin insan benzeri zeka gösterme yeteneğidir. Günümüzde yapay zeka, sağlık, eğitim, "
            "ulaşım ve iletişim gibi birçok alanda hayatımızı değiştirmektedir. Makine öğrenmesi, derin öğrenme ve "
            "doğal dil işleme gibi alt dallar, yapay zekanın temel bileşenleridir. "
            "Sağlık alanında yapay zeka, hastalıkların erken teşhisinde, ilaç geliştirmede ve kişiselleştirilmiş tedavi "
            "planlamada kullanılmaktadır. Radyoloji görüntülerinin analizi, patoloji raporlarının değerlendirilmesi ve "
            "genetik verilerin yorumlanması yapay zekanın sağlıktaki başlıca uygulama alanlarıdır. "
            "Eğitimde yapay zeka, öğrencilerin bireysel öğrenme hızlarına uygun içerik sunarak kişiselleştirilmiş "
            "eğitim deneyimleri yaratmaktadır. Akıllı öğretim sistemleri, öğrencilerin güçlü ve zayıf yönlerini "
            "belirleyerek onlara özel öğrenme yolları çizmektedir. "
            "Ulaşım sektöründe otonom araçlar, trafik yönetim sistemleri ve lojistik optimizasyonu yapay zekanın "
            "önemli uygulama alanlarıdır. Sürücüsüz araçlar, trafik kazalarını azaltma ve ulaşımı daha verimli "
            "hale getirme potansiyeline sahiptir. "
            "Ancak yapay zekanın toplumsal etkileri konusunda ciddi endişeler de bulunmaktadır. İş gücü piyasasında "
            "otomasyon nedeniyle bazı mesleklerin ortadan kalkması, veri gizliliği sorunları, algoritmik önyargılar "
            "ve yapay zekanın kötüye kullanılma riski tartışılan başlıca konulardır. "
            "Yapay zeka etiği, bu teknolojinin sorumlu ve adil bir şekilde geliştirilmesini ve kullanılmasını "
            "sağlamak amacıyla ortaya çıkan bir disiplindir. Şeffaflık, hesap verebilirlik, adalet ve gizlilik "
            "yapay zeka etiğinin temel ilkeleridir."
        ),
        "reference_summary": (
            "Yapay zeka sağlık, eğitim, ulaşım ve iletişim alanlarında hayatımızı değiştirmektedir. "
            "Sağlıkta erken teşhis ve kişiselleştirilmiş tedavi, eğitimde bireysel öğrenme, ulaşımda otonom araçlar "
            "başlıca uygulama alanlarıdır. Ancak iş gücü kaybı, veri gizliliği, algoritmik önyargılar gibi endişeler "
            "mevcuttur. Yapay zeka etiği, şeffaflık, hesap verebilirlik ve adalet ilkeleriyle bu sorunlara çözüm aramaktadır."
        ),
    },
}


def get_sample(key: str) -> dict:
    """Get a sample text by key."""
    return SAMPLES.get(key)


def get_all_samples() -> dict:
    """Get all sample texts."""
    return {k: {"title": v["title"], "language": v["language"], "category": v["category"]} for k, v in SAMPLES.items()}
