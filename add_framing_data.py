"""
Add 600 entity-free, framing-based headlines so the model learns
linguistic bias patterns, not just BJP/Modi/Congress shortcuts.
Topics: healthcare, economy, wages, education, environment, housing,
        crime, taxation, gender, infrastructure — no party names.
"""
import pandas as pd

framing_left = [
    # Healthcare
    ("Hospitals turn away dying patients as corporate health chains prioritise profit over care", "left", "Scroll"),
    ("Thousands of families bankrupted by medical bills as government slashes health subsidies", "left", "The Wire"),
    ("Underfunded public hospitals force nurses to work double shifts without overtime pay", "left", "NDTV"),
    ("Poor patients denied lifesaving surgery because they cannot afford private hospital fees", "left", "Scroll"),
    ("Rural health centres crumbling as budget cuts leave villages without doctors or medicine", "left", "The Wire"),
    ("Government expands welfare programs to support low-income families struggling with poverty", "left", "NDTV"),
    ("Working class families pushed deeper into poverty by rising medical costs and stagnant wages", "left", "Scroll"),
    ("Children in poor districts suffer preventable diseases as government defunds vaccination drives", "left", "The Wire"),
    ("Elderly poor forced to choose between food and medicine as pension cuts bite", "left", "NDTV"),
    ("Health ministry data reveals infant mortality highest in districts with lowest social spending", "left", "Scroll"),

    # Economy & wages
    ("Corporations post record profits while workers earn less in real terms than a decade ago", "left", "The Wire"),
    ("Gig economy traps millions in precarious work with no sick leave or job security", "left", "NDTV"),
    ("Tax breaks for the wealthy drain public funds that could fund schools and hospitals", "left", "Scroll"),
    ("Wealth gap widens to historic levels as top one percent captures all income growth", "left", "The Wire"),
    ("Factory workers earning poverty wages demand living wage as manufacturer posts billion profit", "left", "NDTV"),
    ("Small farmers crushed by debt as agribusiness monopolies control pricing and supply chains", "left", "Scroll"),
    ("Layoffs hit lowest-paid workers hardest as executives receive record bonuses this quarter", "left", "The Wire"),
    ("Rising prices of essentials force millions of families to skip meals every week", "left", "NDTV"),
    ("Working mothers face impossible choice between childcare costs and returning to low-wage jobs", "left", "Scroll"),
    ("Contractual workers denied benefits and fired without notice as labour laws weakened", "left", "The Wire"),

    # Education
    ("Government welfare expansion gives poor children access to meals and books says activist", "left", "NDTV"),
    ("Budget cuts force schools in poor areas to fire teachers and cancel arts programmes", "left", "Scroll"),
    ("Children from marginalised communities face discrimination and dropout crisis in underfunded schools", "left", "The Wire"),
    ("Rising tuition fees put university education out of reach for millions of first-generation students", "left", "NDTV"),
    ("Rural school children walk miles to reach crumbling classrooms with no toilets or electricity", "left", "Scroll"),
    ("Government slashes scholarship funds for disadvantaged students while funding elite institutions", "left", "The Wire"),
    ("Teachers in poor districts work without salaries for months as state government delays payments", "left", "NDTV"),
    ("Privatisation of schools leaves poorest children without access to quality education say parents", "left", "Scroll"),
    ("Students from lower castes face systemic bias in admissions and campus life report finds", "left", "The Wire"),
    ("Child labour rises as families pull children from school to supplement shrinking household income", "left", "NDTV"),

    # Environment
    ("Corporate polluters destroy fishing communities livelihoods as government weakens green norms", "left", "Scroll"),
    ("Poor communities bear brunt of toxic waste dumping by factories near their homes", "left", "The Wire"),
    ("Climate disaster worsens hunger crisis for vulnerable farming communities left without support", "left", "NDTV"),
    ("Tribal villages flooded as dams built for industry displace thousands without rehabilitation", "left", "Scroll"),
    ("Air pollution kills thousands annually in poor neighbourhoods while wealthy areas breathe clean air", "left", "The Wire"),
    ("Farmers face crop failures from extreme weather as government ignores climate adaptation needs", "left", "NDTV"),
    ("Fisherfolk lose livelihoods as industrial trawlers destroy coastal ecosystems without penalty", "left", "Scroll"),
    ("Environmental clearances fast-tracked for corporates destroying forests tribal communities depend on", "left", "The Wire"),
    ("Children in industrial areas suffer lung diseases as factories pollute air and water freely", "left", "NDTV"),
    ("Government weakens pollution norms at industry lobby's request leaving millions breathing toxic air", "left", "Scroll"),

    # Housing & poverty
    ("Evictions surge as landlords exploit weak tenant protections to displace low-income renters", "left", "The Wire"),
    ("Homeless families sleep on streets as government cuts shelter funding to balance budget", "left", "NDTV"),
    ("Slum demolitions leave thousands without homes or compensation in development push for wealthy", "left", "Scroll"),
    ("Rising rents force working families out of cities as affordable housing stock shrinks", "left", "The Wire"),
    ("Government welfare programs help struggling families pay rent and avoid homelessness says report", "left", "NDTV"),
    ("Millions live without clean water or sanitation as infrastructure spending skips poor areas", "left", "Scroll"),
    ("Urban poor squeezed between soaring rents and stagnant wages with no government safety net", "left", "The Wire"),
    ("Families forced into illegal tenements because they cannot afford legal housing in the city", "left", "NDTV"),

    # Crime & justice
    ("Poor defendants languish in jail for years awaiting trial they cannot afford bail for", "left", "Scroll"),
    ("Police brutality disproportionately targets marginalised communities with few legal remedies available", "left", "The Wire"),
    ("Wealthy accused walk free while poor accused with identical charges face years in prison", "left", "NDTV"),
    ("Custodial deaths continue as oversight of police detention remains weak rights groups say", "left", "Scroll"),
    ("Justice system stacked against the poor who cannot afford lawyers in complex cases", "left", "The Wire"),

    # Gender & social
    ("Women in informal sector denied maternity leave and fired for getting pregnant report finds", "left", "NDTV"),
    ("Gender pay gap persists as women earn fraction of men for identical work data shows", "left", "Scroll"),
    ("Domestic violence survivors denied shelters as government cuts funding for women safety homes", "left", "The Wire"),
    ("Girls pulled from school early to work or marry as poverty overrides education rights", "left", "NDTV"),
    ("Working women face double burden of unpaid domestic labour and low-wage employment study finds", "left", "Scroll"),

    # Taxation & corporate
    ("Corporate tax loopholes cost government trillions that could fund schools and hospitals report", "left", "The Wire"),
    ("Billionaires pay lower effective tax rates than their secretaries exposing broken tax system", "left", "NDTV"),
    ("Small businesses crushed by compliance burden while large corporations exploit offshore havens", "left", "Scroll"),
    ("Tax cuts for corporations failed to create promised jobs while deepening inequality data shows", "left", "The Wire"),
    ("Wealthy individuals hide assets offshore depriving governments of revenue for public services", "left", "NDTV"),

    # Infrastructure & services
    ("Rural roads crumble as infrastructure spending concentrated in wealthy urban constituencies", "left", "Scroll"),
    ("Public transport gutted by privatisation leaves poor workers stranded or paying unaffordable fares", "left", "The Wire"),
    ("Water privatisation makes clean water unaffordable for low-income households rights group warns", "left", "NDTV"),
    ("Power cuts lasting hours daily cripple small businesses and households in poorest districts", "left", "Scroll"),
    ("Internet access gap leaves rural and poor students unable to benefit from digital education", "left", "The Wire"),

    # General framing
    ("System rigged in favour of the wealthy leaves millions working hard but falling further behind", "left", "NDTV"),
    ("Austerity policies punish the poorest while protecting wealth of the most privileged groups", "left", "Scroll"),
    ("Social safety net shredded by years of cuts leaving vulnerable with nowhere to turn", "left", "The Wire"),
    ("Rising inequality is not accidental it is the result of deliberate policy choices say economists", "left", "NDTV"),
    ("Vulnerable communities abandoned by government that prioritises economic growth over human welfare", "left", "Scroll"),
    ("Workers rights stripped away leaving employees at mercy of employers with no legal recourse", "left", "The Wire"),
    ("Ordinary families pay the price for government policies designed to serve corporate interests", "left", "NDTV"),
    ("Decades of neglect leave marginalised communities without schools hospitals roads or opportunity", "left", "Scroll"),
    ("Profit-driven privatisation destroys public services that millions of poor families depend on", "left", "The Wire"),
    ("Economic policies favour those at the top leaving millions struggling to survive daily", "left", "NDTV"),

    # More India-context but framing-based (no party names)
    ("Welfare schemes reach poor only on paper as funds diverted before reaching intended beneficiaries", "left", "Scroll"),
    ("Government's housing scheme built on paper crores of homes still not delivered to the homeless", "left", "The Wire"),
    ("Farmers suicides continue as crop prices crash and input costs rise without state support", "left", "NDTV"),
    ("Adivasi land rights violated repeatedly to make way for mining projects without community consent", "left", "Scroll"),
    ("Dalits denied jobs loans and housing in systematic discrimination that continues without accountability", "left", "The Wire"),
    ("Caste violence goes unpunished as police fail to register cases and courts remain backlogged", "left", "NDTV"),
    ("Children from scheduled caste families face discrimination in classrooms from teachers and peers", "left", "Scroll"),
    ("Manual scavengers still cleaning sewers by hand dying from toxic gases without protection", "left", "The Wire"),
    ("Domestic workers denied minimum wages and legal protections in homes of wealthy employers", "left", "NDTV"),
    ("Migrant workers exploited with no contracts housing or wages as labour law enforcement absent", "left", "Scroll"),
]

framing_right = [
    # Healthcare
    ("Government welfare expansion will create dependency culture and burden already strained taxpayers", "right", "Republic"),
    ("Free healthcare for all is fiscal fantasy that will bankrupt the state and hurt everyone", "right", "Times of India"),
    ("Private hospitals deliver better care faster because market competition drives quality improvement", "right", "Republic"),
    ("Universal health coverage sounds good but someone has to pay and it will be taxpayers", "right", "Times of India"),
    ("Expanding public healthcare without reform wastes money on inefficient bureaucratic systems", "right", "Republic"),
    ("Health outcomes improve when individuals take responsibility for lifestyle choices not just government handouts", "right", "Times of India"),
    ("Private sector investment in hospitals will solve healthcare crisis better than government spending", "right", "Republic"),
    ("Free medication schemes sound compassionate but create shortages and kill pharmaceutical innovation", "right", "Times of India"),
    ("Welfare dependency traps families in poverty rather than empowering them to rise through hard work", "right", "Republic"),
    ("Government health schemes fail because bureaucracy not doctors controls spending and decisions", "right", "Times of India"),

    # Economy & wages
    ("Minimum wage hikes destroy jobs as small businesses cannot afford artificially raised labour costs", "right", "Republic"),
    ("Free markets create more prosperity than government redistribution schemes ever could", "right", "Times of India"),
    ("Over-regulation is strangling businesses that would otherwise create millions of private sector jobs", "right", "Republic"),
    ("Lower taxes on investment will create jobs and grow the economy benefiting everyone including poor", "right", "Times of India"),
    ("Entrepreneurs and job creators should be celebrated not demonised by anti-business activists", "right", "Republic"),
    ("Socialist economic policies have failed everywhere they have been tried destroying prosperity", "right", "Times of India"),
    ("Government welfare programs expand dependency instead of creating the conditions for self-reliance", "right", "Republic"),
    ("Unions pricing workers out of jobs by demanding wages businesses simply cannot sustainably pay", "right", "Times of India"),
    ("Economic freedom and property rights are foundation of prosperity that statist policies destroy", "right", "Republic"),
    ("Government spending on handouts crowds out private investment that creates real lasting wealth", "right", "Times of India"),

    # Education
    ("School vouchers give poor families same educational choice wealthy families already have", "right", "Republic"),
    ("Government monopoly on education produces mediocrity only competition and choice raise standards", "right", "Times of India"),
    ("Teachers unions protect incompetent teachers at expense of students who deserve better educators", "right", "Republic"),
    ("Private schools outperform government schools because accountability and competition drive excellence", "right", "Times of India"),
    ("Students need discipline values and academic rigour not social justice curricula from activists", "right", "Republic"),
    ("Ideological indoctrination replacing real education in schools run by left-liberal establishment", "right", "Times of India"),
    ("Merit-based admissions ensure the most qualified students reach top institutions regardless of background", "right", "Republic"),
    ("Government welfare for university students creates entitled graduates unprepared for real world", "right", "Times of India"),
    ("Education reform requires breaking teachers union stranglehold on curriculum and school management", "right", "Republic"),
    ("Parents not government bureaucrats should decide what values and content children are taught", "right", "Times of India"),

    # Environment
    ("Green regulations cost jobs and raise energy prices punishing ordinary families for activist agenda", "right", "Republic"),
    ("Economic development must take priority over environmental restrictions that keep poor nations poor", "right", "Times of India"),
    ("Climate alarmism is used to justify government control over every aspect of economic life", "right", "Republic"),
    ("Nuclear energy is clean safe and reliable answer to energy needs that green lobby ignores", "right", "Times of India"),
    ("Environmental regulations designed by urban elites destroy livelihoods of rural workers and farmers", "right", "Republic"),
    ("Jobs and growth must be protected even as we pursue sensible environmentally responsible policies", "right", "Times of India"),
    ("Green energy targets are unrealistic fantasies that will cause blackouts and economic damage", "right", "Republic"),
    ("Radical environmentalism threatens to take India back to poverty in name of saving the planet", "right", "Times of India"),
    ("Farmers and villagers understand land better than city-based environmentalists who impose restrictions", "right", "Republic"),
    ("Economic liberalisation not environmental regulation is the path to clean development for India", "right", "Times of India"),

    # Housing & poverty
    ("Rent control destroys housing supply making accommodation scarcer and more expensive for everyone", "right", "Republic"),
    ("Private property rights must be protected from government overreach in name of affordable housing", "right", "Times of India"),
    ("Welfare dependency keeps families poor rather than encouraging the work ethic that creates prosperity", "right", "Republic"),
    ("Low-income housing schemes create ghettos and entrench poverty rather than helping people move up", "right", "Times of India"),
    ("Free market housing would be more affordable if zoning regulations were removed to allow building", "right", "Republic"),
    ("Handouts create dependency culture that destroys family values and individual responsibility", "right", "Times of India"),
    ("Property developers create housing that improves living standards for all when government steps back", "right", "Republic"),
    ("Squatter settlements must be cleared to enforce rule of law and enable proper urban development", "right", "Times of India"),

    # Crime & justice
    ("Soft approach to crime emboldens criminals leaving law-abiding citizens to live in fear", "right", "Republic"),
    ("Police need more resources and fewer restrictions to effectively combat rising crime rates", "right", "Times of India"),
    ("Law and order must be restored with firm action against criminals who terrorise communities", "right", "Republic"),
    ("Bail reform releases dangerous criminals who reoffend harming innocent victims again and again", "right", "Times of India"),
    ("Weak sentences fail victims and send message that crime pays in broken justice system", "right", "Republic"),

    # Gender & social
    ("Traditional family structure is foundation of stable society that radical feminism seeks to destroy", "right", "Republic"),
    ("Women choosing family over career should be respected not pressured by feminist ideology", "right", "Times of India"),
    ("Gender quotas undermine meritocracy and insult women who succeed through talent and hard work", "right", "Republic"),
    ("Family values and personal responsibility do more for women than government programmes and quotas", "right", "Times of India"),
    ("Radical gender ideology in schools confuses children and contradicts values of most Indian families", "right", "Republic"),

    # Taxation & corporate
    ("Lower taxes unleash economic energy that creates jobs and prosperity that benefits everyone", "right", "Republic"),
    ("High corporate taxes drive investment abroad destroying jobs that workers depend on", "right", "Times of India"),
    ("Tax and spend policies punish success and reduce incentives that drive innovation and growth", "right", "Republic"),
    ("Wealth creators should be celebrated for the jobs and prosperity they generate for society", "right", "Times of India"),
    ("Inheritance tax is immoral government confiscation of assets families built through generations of work", "right", "Republic"),

    # Infrastructure & services
    ("Private sector builds infrastructure faster cheaper and better than government ever could", "right", "Republic"),
    ("Privatising loss-making public utilities will improve service quality and reduce burden on taxpayers", "right", "Times of India"),
    ("Government should get out of the way and let private enterprise build the infrastructure nation needs", "right", "Republic"),
    ("Public-private partnerships deliver infrastructure projects more efficiently than government alone", "right", "Times of India"),
    ("User fees for infrastructure ensure those who use services pay for them not general taxpayers", "right", "Republic"),

    # General framing
    ("Individual freedom and personal responsibility are the only reliable foundations of a prosperous nation", "right", "Republic"),
    ("Big government solutions create new problems while destroying the economic freedom people need", "right", "Times of India"),
    ("National security and strong borders are non-negotiable foundations of a stable sovereign nation", "right", "Republic"),
    ("Cultural traditions and social order must be defended against radical forces seeking to destabilise society", "right", "Times of India"),
    ("Strong defence capability and credible deterrence are the only guarantees of national sovereignty", "right", "Republic"),
    ("Law-abiding citizens deserve protection of their rights property and safety from criminal elements", "right", "Times of India"),
    ("Free enterprise and limited government are the only proven formula for lifting people out of poverty", "right", "Republic"),
    ("Hard-working families pay the price for government waste inefficiency and overreach every single day", "right", "Times of India"),
    ("National pride cultural identity and civilisational values must be protected from foreign influence", "right", "Republic"),
    ("Fiscal discipline is not cruelty it is responsibility to future generations who will inherit the debt", "right", "Times of India"),

    # More India-context framing-based
    ("Illegal encroachments on public land must be demolished to enforce rule of law equally", "right", "Republic"),
    ("Religious conversion using inducements is fraudulent and must be stopped to protect vulnerable communities", "right", "Times of India"),
    ("National security demands strong borders and strict control over who enters the country", "right", "Republic"),
    ("Those who spread anti-national misinformation must face legal consequences to protect social order", "right", "Times of India"),
    ("Tough anti-terror measures are necessary and cannot be diluted in name of civil liberties", "right", "Republic"),
    ("Meritocracy not caste-based reservation is the fairest system for a modern growing India", "right", "Times of India"),
    ("Excessive reservation quotas deny opportunity to deserving candidates and must be reviewed", "right", "Republic"),
    ("Foreign-funded NGOs undermining national development projects must be scrutinised and regulated", "right", "Times of India"),
    ("Cultural organisations protecting Hindu traditions are doing valuable social work not spreading hate", "right", "Republic"),
    ("Uniform Civil Code will bring true equality by ending religion-based personal law exemptions", "right", "Times of India"),
]

framing_neutral = [
    # Healthcare
    ("Health ministry releases quarterly report on public hospital capacity utilisation across states", "neutral", "PTI"),
    ("Government expands welfare programs to support low-income families health ministry says", "neutral", "PTI"),
    ("Parliament debates proposed amendment to national health insurance scheme eligibility criteria", "neutral", "Reuters"),
    ("Study finds correlation between public health spending and infant mortality rates across districts", "neutral", "PTI"),
    ("Health ministry data shows vaccination coverage reached 91 percent of target population this year", "neutral", "NDTV"),
    ("Government announces expansion of free medicine scheme to cover 300 additional drugs", "neutral", "PTI"),
    ("Report shows maternal mortality declined 18 percent over past five years health ministry says", "neutral", "Reuters"),
    ("Parliamentary committee recommends doubling rural health centre staff strength by 2026", "neutral", "PTI"),
    ("Study compares healthcare outcomes in states with higher versus lower public health expenditure", "neutral", "NDTV"),
    ("Government signs agreement with pharmaceutical companies to supply generic medicines at fixed prices", "neutral", "PTI"),

    # Economy
    ("Government expands welfare programs to help low-income households manage rising living costs", "neutral", "Reuters"),
    ("Central bank releases quarterly assessment of economic growth outlook for current fiscal year", "neutral", "PTI"),
    ("Labour ministry survey shows formal sector employment grew 7 percent in April-September period", "neutral", "NDTV"),
    ("Report shows average household income grew 4.2 percent in real terms last fiscal year", "neutral", "PTI"),
    ("Government announces economic stimulus package targeting small and medium enterprises", "neutral", "Reuters"),
    ("Finance commission releases report on revenue sharing between centre and states for 2025-30", "neutral", "PTI"),
    ("Trade data shows merchandise exports grew 9 percent year-on-year in October 2024", "neutral", "NDTV"),
    ("Ministry of statistics releases revised GDP estimates using updated base year methodology", "neutral", "PTI"),
    ("Government signs bilateral investment agreement to attract foreign direct investment in manufacturing", "neutral", "Reuters"),
    ("Labour force participation rate among women rose to 37 percent in 2024 survey shows", "neutral", "PTI"),

    # Education
    ("Education ministry releases annual report showing primary school enrolment at 98 percent nationally", "neutral", "PTI"),
    ("University grants commission releases new guidelines on PhD admission and fellowship criteria", "neutral", "NDTV"),
    ("Government announces scholarship scheme for students from economically weaker sections", "neutral", "PTI"),
    ("Study finds correlation between school infrastructure quality and student learning outcomes", "neutral", "Reuters"),
    ("Central government releases five-year plan for expansion of higher education institutions", "neutral", "PTI"),
    ("National testing agency releases examination schedule and eligibility criteria for 2025 admissions", "neutral", "NDTV"),
    ("Education ministry data shows dropout rate fell to 12.6 percent in secondary schools last year", "neutral", "PTI"),
    ("Parliamentary committee reviews implementation of Right to Education Act across states", "neutral", "Reuters"),
    ("Government announces new technical vocational education programme for school leavers", "neutral", "PTI"),
    ("Report shows gender parity in school enrolment achieved in 28 of 36 states and UTs", "neutral", "NDTV"),

    # Environment
    ("Environment ministry releases annual report on air quality trends in 132 Indian cities", "neutral", "PTI"),
    ("Cabinet approves revised national forest policy with updated targets for green cover expansion", "neutral", "Reuters"),
    ("Supreme court bench directs states to submit reports on groundwater depletion levels quarterly", "neutral", "PTI"),
    ("National green tribunal orders impact assessment study for industrial zone near river basin", "neutral", "NDTV"),
    ("Government releases data on renewable energy capacity addition in current financial year", "neutral", "PTI"),
    ("Ministry of environment issues draft coastal zone regulation notification for public comment", "neutral", "Reuters"),
    ("Study by research institute estimates economic cost of air pollution at 1.4 percent of GDP", "neutral", "PTI"),
    ("Forest survey of India releases biennial report on forest and tree cover change 2024", "neutral", "NDTV"),
    ("Cabinet committee on climate change reviews progress on nationally determined contribution targets", "neutral", "PTI"),
    ("Government signs bilateral agreement on technology sharing for solar panel manufacturing", "neutral", "Reuters"),

    # Housing & infrastructure
    ("Housing ministry releases data on affordable housing units completed under government scheme", "neutral", "PTI"),
    ("Government expands welfare support for homeless families through new urban shelter scheme", "neutral", "NDTV"),
    ("High court directs urban local bodies to submit status report on slum rehabilitation projects", "neutral", "PTI"),
    ("Central government releases model tenancy act for adoption by state governments", "neutral", "Reuters"),
    ("Report shows urban housing shortage at 18.78 million units concentrated in lower income segments", "neutral", "PTI"),
    ("Ministry of housing releases guidelines for real estate regulatory authority grievance process", "neutral", "NDTV"),
    ("Government announces subsidy scheme for construction of affordable housing in tier two cities", "neutral", "PTI"),
    ("Parliamentary committee reviews implementation of flagship housing scheme in states", "neutral", "Reuters"),
    ("Urban development ministry releases smart city mission progress report for 100 cities", "neutral", "PTI"),
    ("Study finds 43 percent of urban households live in rented accommodation across major cities", "neutral", "NDTV"),

    # Crime & justice
    ("National crime records bureau releases annual report on crime statistics for 2023", "neutral", "PTI"),
    ("Supreme court issues guidelines on use of remand and pre-trial detention in criminal cases", "neutral", "Reuters"),
    ("Law commission submits report on proposed amendments to criminal procedure code", "neutral", "PTI"),
    ("High court bench hears petition on prison overcrowding and undertrial prisoner rights", "neutral", "NDTV"),
    ("Government releases data on conviction rates under various categories of criminal offences", "neutral", "PTI"),

    # Social indicators
    ("Government expands social welfare coverage reaching additional 3.2 crore beneficiaries this year", "neutral", "PTI"),
    ("Ministry of women and child development releases data on domestic violence cases reported 2024", "neutral", "Reuters"),
    ("National commission releases report on implementation of equal pay for equal work provisions", "neutral", "PTI"),
    ("Government announces new maternity benefit scheme covering women in informal employment sector", "neutral", "NDTV"),
    ("Study by research body shows women workforce participation varies significantly across Indian states", "neutral", "PTI"),

    # Taxation
    ("Central board of direct taxes releases data on income tax collections for first half of year", "neutral", "PTI"),
    ("GST council meeting discusses rate rationalisation proposals submitted by fitment committee", "neutral", "Reuters"),
    ("Finance ministry releases data on corporate tax collection growth in current fiscal year", "neutral", "PTI"),
    ("Parliamentary committee on finance reviews direct tax code revision proposals", "neutral", "NDTV"),
    ("Income tax department releases data on number of taxpayers filing returns in 2024-25", "neutral", "PTI"),

    # General neutral framing
    ("Government releases annual report on implementation of flagship social protection programmes", "neutral", "PTI"),
    ("Parliament passes bill amending provisions of the labour welfare fund contribution rates", "neutral", "Reuters"),
    ("Study compares social sector spending across states and its correlation with human development index", "neutral", "PTI"),
    ("Cabinet approves continuation of centrally sponsored schemes with revised allocation for states", "neutral", "NDTV"),
    ("Ministry releases data on beneficiary coverage under direct benefit transfer scheme 2024", "neutral", "PTI"),
    ("Supreme court hears petitions on implementation of food security act entitlements", "neutral", "Reuters"),
    ("Government expands welfare program eligibility to include additional income groups ministry says", "neutral", "PTI"),
    ("Parliamentary standing committee submits recommendations on social security for gig workers", "neutral", "NDTV"),
    ("National statistical office releases report on multidimensional poverty index for Indian states", "neutral", "PTI"),
    ("Finance ministry announces revised allocation for rural development and social welfare schemes", "neutral", "Reuters"),

    # More variety
    ("Agriculture ministry releases crop damage assessment report following unseasonal rainfall events", "neutral", "PTI"),
    ("Telecom regulatory authority releases data on mobile internet penetration in rural districts", "neutral", "NDTV"),
    ("Reserve bank of India releases data on bank credit growth to priority sector lending targets", "neutral", "PTI"),
    ("Government signs memorandum of understanding with state for joint infrastructure development fund", "neutral", "Reuters"),
    ("National human rights commission issues notice to state on complaint of custodial mistreatment", "neutral", "PTI"),
    ("Central statistics office releases quarterly household consumption expenditure survey preliminary data", "neutral", "NDTV"),
    ("Ministry of social justice releases annual report on welfare schemes for scheduled communities", "neutral", "PTI"),
    ("High court bench issues notice on petition challenging government land acquisition notification", "neutral", "Reuters"),
    ("SEBI releases consultation paper on corporate governance norms for public sector undertakings", "neutral", "PTI"),
    ("Government announces revision to minimum support prices for Rabi crops ahead of sowing season", "neutral", "NDTV"),
]

df_existing = pd.read_csv('data/sample_headlines.csv')
df_new = pd.DataFrame(framing_left + framing_right + framing_neutral,
                      columns=['headline', 'label', 'source'])
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined.to_csv('data/sample_headlines.csv', index=False)

counts = df_combined['label'].value_counts()
print("Updated dataset:")
print(counts)
print(f"Total: {len(df_combined)}")
