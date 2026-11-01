import { useState } from 'react'
import './App.css'
import profilePlaceholder from './assets/my_image_1.png'
import sqlIcon from './assets/icons/sql.png'
import azureIcon from './assets/icons/azure.png'
import vertexaiIcon from './assets/icons/vertexai.png'
import langfuseIcon from './assets/icons/langfuse.png'
import langchainIcon from './assets/icons/langchain.png'
import langgraphIcon from './assets/icons/langgraph.png'
import crewaiIcon from './assets/icons/crewai.png'
import microsoftIcon from './assets/icons/microsoft.png'
import deloitteIcon from './assets/icons/deloitte.png'
import {
  FaEnvelope, FaLinkedin, FaGithub,
  FaUser, FaBolt, FaBriefcase, FaGraduationCap,
  FaCertificate, FaTrophy, FaAws, FaAward, FaCode, FaBookOpen, FaFileDownload, FaServer, FaDatabase, FaStream,
} from 'react-icons/fa'
import { SiGooglebigquery, SiMongodb, SiRedis, SiApachekafka, SiApacheairflow, SiWarnerbros, SiLeetcode, SiPython, SiApachespark, SiGooglecloud, SiDatabricks, SiKubernetes, SiRedhatopenshift, SiFastapi, SiMlflow, SiPytorch, SiTensorflow, SiDocker, SiGithubactions, SiHelm, SiArgo, SiScikitlearn } from 'react-icons/si'

const RESUME_FILENAME = 'Mayur_G_D_CV.pdf'
const RESUME_URL = `${import.meta.env.BASE_URL}${RESUME_FILENAME}`

const DAY_MS = 1000 * 60 * 60 * 24
const CAREER_START = new Date('2020-08-17')
const yearsSince = (d) => Math.floor((new Date() - d) / (DAY_MS * 365))

const TECH_STACK = [
  {
    category: 'Programming Languages',
    accent: '#f5c518',
    items: [
      { name: 'Python',  Icon: SiPython },
      { name: 'SQL', iconSrc: sqlIcon },
      { name: 'PySpark', Icon: SiApachespark },
    ],
  },
  {
    category: 'Cloud & Infrastructure',
    accent: '#3b82f6',
    items: [
      { name: 'GCP',        Icon: SiGooglecloud },
      { name: 'BigQuery',   Icon: SiGooglebigquery },
      { name: 'Cloud Run',  Icon: FaServer },
      { name: 'Vertex AI', iconSrc: vertexaiIcon },
      { name: 'Databricks', Icon: SiDatabricks },
      { name: 'Azure', iconSrc: azureIcon },
      { name: 'Kubernetes', Icon: SiKubernetes },
      { name: 'OpenShift',  Icon: SiRedhatopenshift },
      { name: 'AWS',        Icon: FaAws },
    ],
  },
  {
    category: 'AI / ML',
    accent: '#ef4444',
    chipStyle: true,
    tall: true,
    items: [
      { name: 'Classical ML' },
      { name: 'Deep Learning' },
      { name: 'Time Series Forecasting' },
      { name: 'NLP' },
      { name: 'LLMs & RAG' },
      { name: 'Agentic AI' },
      { name: 'Multimodal / Video Understanding' },
      { name: 'Speech & Audio AI (TTS, ASR)' },
      { name: 'MCP' },
    ],
  },
  {
    category: 'Frameworks & Libraries',
    accent: '#22c55e',
    items: [
      { name: 'PyTorch',     Icon: SiPytorch },
      { name: 'MLflow',      Icon: SiMlflow },
      { name: 'scikit-learn', Icon: SiScikitlearn },
      { name: 'TensorFlow',  Icon: SiTensorflow },
      { name: 'FastAPI',     Icon: SiFastapi },
      { name: 'Langfuse', iconSrc: langfuseIcon },
    ],
  },
  {
    category: 'Agent Frameworks',
    accent: '#a855f7',
    items: [
      { name: 'LangGraph', iconSrc: langgraphIcon },
      { name: 'LangChain', iconSrc: langchainIcon },
      { name: 'CrewAI', iconSrc: crewaiIcon },
      { name: 'MS Agent Framework', iconSrc: microsoftIcon },
    ],
  },
  {
    category: 'Data & Vector Stores',
    accent: '#ec4899',
    items: [
      { name: 'ChromaDB',      Icon: FaDatabase },
      { name: 'MongoDB Atlas', Icon: SiMongodb },
      { name: 'Redis',         Icon: SiRedis },
      { name: 'Kafka',         Icon: SiApachekafka },
    ],
  },
  {
    category: 'MLOps & Orchestration',
    accent: '#f97316',
    items: [
      { name: 'Metaflow',       Icon: FaStream },
      { name: 'Airflow',        Icon: SiApacheairflow },
      { name: 'Docker',         Icon: SiDocker },
      { name: 'GitHub Actions', Icon: SiGithubactions },
      { name: 'Helm',           Icon: SiHelm },
      { name: 'Argo',           Icon: SiArgo },
    ],
  },
]

const NAV = [
  { id: 'home',           Icon: FaUser,          label: 'Home'           },
  { id: 'experience',     Icon: FaBriefcase,     label: 'Experience'     },
  { id: 'skills',         Icon: FaBolt,          label: 'Tech Stack'     },
  { id: 'awards',         Icon: FaTrophy,        label: 'Awards'         },
  { id: 'certifications', Icon: FaCertificate,   label: 'Certifications' },
  { id: 'education',      Icon: FaGraduationCap, label: 'Education'      },
]

const CAPTIONS = {
  home:           'Senior Machine Learning Engineer · Bengaluru, India',
  skills:         'Tech Stack',
  experience:     'Experience',
  education:      'Education',
  certifications: 'Certifications',
  awards:         'Awards & Recognition',
}

/* ── Tech Stack section ── */
function TechStack() {
  return (
    <div className="techstack-wrapper">
      <div className="techstack">
        {TECH_STACK.map(({ category, accent, chipStyle, tall, items }) => (
          <div key={category} className="techstack-col" style={{ '--cat-accent': accent, gridRow: tall ? 'span 2' : undefined }}>
            <div className="techstack-col-header">
              <span className="techstack-col-header-text">{category}</span>
            </div>
            {chipStyle ? (
              <div className="techstack-chip-grid">
                {items.map(({ name }) => (
                  <span key={name} className="techstack-chip">{name}</span>
                ))}
              </div>
            ) : (
              <ul className="techstack-list">
                {items.map(({ name, Icon, iconSrc }) => (
                  <li key={name} className="techstack-item">
                    {Icon && <Icon className="techstack-item-icon" />}
                    {iconSrc && <img src={iconSrc} alt={name} className="techstack-item-icon" />}
                    <span>{name}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Experience section ── */
const COMPANIES = [
  {
    name: 'WARNER BROS.',
    suffix: 'DISCOVERY',
    short: 'WBD',
    location: 'Bengaluru · Media & Entertainment',
    Logo: SiWarnerbros,
    from: new Date('2026-04-20'),
    to:   null,
    roles: [
      {
        title: 'Senior ML Engineer',
        period: 'Apr 2026 — Present',
        accent: '#a855f7',
        points: [
          'Built an end-to-end generative AI pipeline that writes audio description scripts for films and series, using Gemini video understanding with speech and scene analysis to find and narrate the gaps between dialogue.',
          'Took it to production on GCP for feature-length titles, with evaluation against human-written scripts and cost tracking per title.',
          'Built the initial text-to-speech prototype that narrates AD scripts with Gemini TTS, including a narrator-voice profiler and an automated quality framework (timing fit, word accuracy, naturalness, voice consistency) whose checks carried into the production service.',
        ],
      },
    ],
  },
  {
    name: 'DELOITTE',
    suffix: 'USI',
    short: 'DELOITTE',
    location: 'Bengaluru · Consulting for US clients',
    logoSrc: deloitteIcon,
    from: new Date('2020-08-17'),
    to:   new Date('2026-04-10'),
    roles: [
      {
        title: 'Consultant — Data Scientist',
        period: 'Jun 2023 — Apr 2026',
        accent: '#3b82f6',
        points: [
          'Owned the demand-forecasting solver integration through MVP and built a new forecasting model that significantly improved forecast accuracy and efficiency.',
          'Built agentic AI applications with CrewAI and LangGraph for AIOps and infrastructure automation.',
          'Developed reusable Databricks MLOps stack templates for model training, deployment and job orchestration.',
        ],
      },
      {
        title: 'Analyst — Data Scientist',
        period: 'Aug 2020 — May 2023',
        accent: '#22c55e',
        points: [
          'Cut processing time by more than 10x across key sections of a clustering pipeline and became the team expert on its clustering models.',
          'Led the migration of model code to PySpark and set up MLflow experiment tracking and scheduled jobs, improving scalability and run traceability.',
        ],
      },
    ],
  },
]

function tenure(from, to) {
  const ms = (to ?? new Date()) - from
  return {
    yrs: Math.floor(ms / (DAY_MS * 365)),
    mos: Math.floor((ms / (DAY_MS * 30.44)) % 12),
  }
}

function Experience() {
  return (
    <div className="exp-wrapper">
      {COMPANIES.map(({ name, suffix, short, location, Logo, logoSrc, from, to, roles }) => {
        const { yrs, mos } = tenure(from, to)
        return (
          <div key={name} className="exp-company">
            <div className="exp-splash">
              <div className="exp-splash-left">
                <div className="exp-splash-eyebrow">
                  {from.getFullYear()} — {to ? to.getFullYear() : 'PRESENT'}
                </div>
                <div className="exp-splash-company">{name} <span className="exp-splash-usi">{suffix}</span></div>
                <div className="exp-splash-tagline">{location}</div>
              </div>
              <div className="exp-splash-center">
                {Logo && <Logo className="exp-splash-logo" aria-label={`${name} ${suffix}`} />}
                {logoSrc && <img src={logoSrc} alt={`${name} ${suffix}`} className="exp-splash-logo" />}
              </div>
              <div className="exp-splash-right">
                <div className="exp-splash-stat">
                  <span className="exp-splash-stat-val">{yrs}</span>
                  <span className="exp-splash-stat-unit">YRS</span>
                </div>
                <div className="exp-splash-divider" />
                <div className="exp-splash-stat">
                  <span className="exp-splash-stat-val">{mos}</span>
                  <span className="exp-splash-stat-unit">MOS</span>
                </div>
                <div className="exp-splash-stat-label">AT<br/>{short}</div>
              </div>
            </div>

            <div className="exp-panels">
              {roles.map(({ title, period, accent, points }, i) => (
                <div key={title} className="exp-timeline-row">
                  <div className="exp-timeline-spine">
                    <div className="exp-timeline-dot" style={{ background: accent, borderColor: accent }} />
                    {i < roles.length - 1 && <div className="exp-timeline-line" style={{ background: `linear-gradient(to bottom, ${accent}, var(--grey-light))` }} />}
                  </div>
                  <div className="exp-panel" style={{ '--role-accent': accent }}>
                    <div className="exp-panel-title">{title}</div>
                    <div className="exp-panel-period">{period}</div>
                    {points.length > 0 && (
                      <ul className="exp-panel-points">
                        {points.map((p) => <li key={p}>{p}</li>)}
                      </ul>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ── Education section ── */
const EDUCATION = [
  {
    id: 'pes',
    institution: 'PES University',
    degree: 'B.Tech — Electronics & Communication Engineering',
    period: '2016 — 2020',
    accent: '#3b82f6',
    tag: 'BACHELOR OF TECHNOLOGY',
    stats: [
      { label: 'CGPA', value: '9.24 / 10' },
    ],
  },
  {
    id: 'jain',
    institution: 'Jain College',
    degree: 'Pre-University Course — PCME',
    period: '2014 — 2016',
    accent: '#22c55e',
    tag: 'PRE-UNIVERSITY',
    stats: [
      { label: 'SCORE', value: '95.83%' },
    ],
  },
  {
    id: 'school',
    institution: 'Sadhguru Sainath International School',
    degree: 'Secondary School',
    period: 'Graduated 2014',
    accent: '#f97316',
    tag: 'SECONDARY',
    stats: [
      { label: 'CGPA', value: '10.0 / 10' },
    ],
  },
]

function Education() {
  return (
    <div className="edu-wrapper">
      <div className="edu-timeline">
        {EDUCATION.map(({ id, institution, degree, period, accent, tag, stats }, idx) => (
          <div key={id} className="edu-entry" style={{ '--edu-accent': accent }}>
            <div className="edu-spine">
              <div className="edu-spine-dot" />
              {idx < EDUCATION.length - 1 && <div className="edu-spine-line" />}
            </div>
            <div className="edu-card">
              <div className="edu-card-header">
                <span className="edu-tag">{tag}</span>
              </div>
              <div className="edu-institution">{institution}</div>
              <div className="edu-degree">{degree}</div>
              <div className="edu-footer">
                <span className="edu-period">{period}</span>
                {stats.length > 0 && (
                  <div className="edu-stats">
                    {stats.map(({ label, value }) => (
                      <div key={label} className="edu-stat">
                        <span className="edu-stat-value">{value}</span>
                        <span className="edu-stat-label">{label}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── Certifications section ── */
const CERTS = [
  {
    id: 'dbx-ml',
    title: 'Databricks Certified Machine Learning Associate',
    issuer: 'Databricks',
    issued: 'Feb 2026',
    expires: 'Feb 2028',
    accent: '#ef4444',
    tag: 'ACTIVE',
    Icon: SiDatabricks,
    level: 'ASSOCIATE',
  },
  {
    id: 'dbx-genai',
    title: 'Databricks Certified Generative AI Engineer Associate',
    issuer: 'Databricks',
    issued: 'Dec 2025',
    expires: 'Dec 2027',
    accent: '#f97316',
    tag: 'ACTIVE',
    Icon: SiDatabricks,
    level: 'ASSOCIATE',
  },
  {
    id: 'ml-spec',
    title: 'Machine Learning Specialization',
    issuer: 'DeepLearning.AI · Stanford University',
    issued: 'Apr 2024',
    expires: null,
    accent: '#3b82f6',
    tag: 'COMPLETED',
    Icon: FaGraduationCap,
    level: 'SPECIALIZATION',
  },
  {
    id: 'dsa',
    title: 'Data Structures & Algorithms in Python',
    issuer: 'Coding Ninjas',
    issued: 'Mar 2020',
    expires: null,
    accent: '#22c55e',
    tag: 'COMPLETED',
    Icon: FaCode,
    level: 'COMPLETION',
  },
]

const COURSERA_CERTS = [
  { title: 'Sequences, Time Series & Prediction',            year: '2023' },
  { title: 'GCP Big Data & ML Fundamentals',                 year: '2023' },
  { title: 'Introduction to Data Science in Python',         year: '2022' },
  { title: 'Introduction to Git & GitHub',                   year: '2022' },
  { title: 'Neural Networks & Deep Learning',                year: '2021' },
  { title: 'Machine Learning',                               year: '2021' },
  { title: 'Using Python to Interact with the OS',           year: '2021' },
  { title: 'Programming for Everybody (Python)',             year: '2021' },
]

const COURSERA_ACCENT = '#0056d2'

function CertSpine({ accent, Icon, withLine }) {
  return (
    <div className="cert-spine">
      <div className="cert-spine-icon" style={{ background: accent, borderColor: accent }}>
        <Icon className="cert-spine-emoji" />
      </div>
      {withLine && <div className="cert-spine-line" style={{ background: `linear-gradient(to bottom, ${accent}, var(--grey-light))` }} />}
    </div>
  )
}

function CertRow({ title, issuer, issued, expires, accent, tag, Icon, level, withLine }) {
  return (
    <div className="cert-row">
      <CertSpine accent={accent} Icon={Icon} withLine={withLine} />
      <div className="cert-card" style={{ '--cert-accent': accent }}>
        <div className="cert-card-top">
          <div className="cert-left">
            <span className="cert-level">{level}</span>
            <span className="cert-title">{title}</span>
            <span className="cert-issuer">{issuer}</span>
          </div>
          <div className="cert-right">
            <span className="cert-tag">{tag}</span>
            <div className="cert-dates">
              <div className="cert-date-item">
                <span className="cert-date-label">ISSUED</span>
                <span className="cert-date-val">{issued}</span>
              </div>
              <div className="cert-date-item">
                <span className="cert-date-label">EXPIRES</span>
                <span className={`cert-date-val${expires ? '' : ' cert-date-never'}`}>{expires ?? 'No expiry'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function Certifications() {
  const topCerts = CERTS.slice(0, 3)
  const lastCert = CERTS[3]
  return (
    <div className="cert-wrapper">
      <div className="cert-timeline">
        {topCerts.map((c) => <CertRow key={c.id} {...c} withLine />)}

        <div className="cert-row">
          <CertSpine accent={COURSERA_ACCENT} Icon={FaBookOpen} withLine />
          <div className="cert-card" style={{ '--cert-accent': COURSERA_ACCENT }}>
            <div className="cert-card-top">
              <div className="cert-left">
                <span className="cert-level">2021 — 2023</span>
                <span className="cert-title">Continuous Learning</span>
                <span className="cert-issuer">Coursera — Online Certifications</span>
              </div>
              <div className="cert-right">
                <span className="cert-tag">{COURSERA_CERTS.length} COURSES</span>
              </div>
            </div>
            <div className="cert-courses-grid">
              {COURSERA_CERTS.map(({ title, year }) => (
                <div key={title} className="cert-course-item">
                  <span className="cert-course-year">{year}</span>
                  <span className="cert-course-name">{title}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <CertRow key={lastCert.id} {...lastCert} withLine={false} />
      </div>
    </div>
  )
}

/* ── Awards section ── */
const AWARDS = [
  {
    id: 'applause-2025',
    level: 'APPLAUSE AWARD',
    title: 'OKR Delivery & Quality',
    year: '2025',
    accent: '#22c55e',
    Icon: FaAward,
    quote: 'Hard work and contribution to the project resulted in completion of committed OKRs and quality delivery.',
  },
  {
    id: 'applause-2024',
    level: 'APPLAUSE AWARD',
    title: 'Forecasting Model & DW Solver',
    year: '2024',
    accent: '#a855f7',
    Icon: FaAward,
    quote: 'Took the majority of responsibilities for the DW solver throughout integration and implemented a new DS model that drastically improved forecasting efficiency and reduced errors.',
  },
  {
    id: 'applause-2023',
    level: 'APPLAUSE AWARD',
    title: 'PySpark Migration & MLOps',
    year: '2023',
    accent: '#3b82f6',
    Icon: FaAward,
    quote: 'Took the lead in converting our code to PySpark, setting up model tracking and creating jobs — improving the scalability and organization of our models.',
  },
  {
    id: 'opa-2022',
    level: 'OUTSTANDING PERFORMANCE AWARD',
    title: '10x Pipeline Optimization',
    year: '2022',
    accent: '#f5c518',
    Icon: FaTrophy,
    quote: 'Reduced processing time by more than 10x in many sections of our code, significantly improved code quality and organization, and became the expert in several of our clustering models.',
  },
]

function Awards() {
  return (
    <div className="exp-wrapper">
      <div className="exp-company">
        <div className="exp-splash">
          <div className="exp-splash-left">
            <div className="exp-splash-eyebrow">2022 — 2025</div>
            <div className="exp-splash-company">DELOITTE <span className="exp-splash-usi">USI</span></div>
            <div className="exp-splash-tagline">Performance recognition</div>
          </div>
          <div className="exp-splash-center">
            <img src={deloitteIcon} alt="Deloitte USI" className="exp-splash-logo" />
          </div>
          <div className="exp-splash-right">
            <div className="exp-splash-stat">
              <span className="exp-splash-stat-val">{AWARDS.length}</span>
            </div>
            <div className="exp-splash-stat-label">AWARDS</div>
          </div>
        </div>

        <div className="cert-timeline">
          {AWARDS.map(({ id, level, title, year, accent, Icon, quote }, i) => (
            <div key={id} className="cert-row">
              <div className="cert-spine">
                <div className="cert-spine-icon" style={{ background: accent, borderColor: accent }}>
                  <Icon className="cert-spine-emoji" />
                </div>
                {i < AWARDS.length - 1 && <div className="cert-spine-line" style={{ background: `linear-gradient(to bottom, ${accent}, var(--grey-light))` }} />}
              </div>
              <div className="cert-card" style={{ '--cert-accent': accent }}>
                <div className="cert-card-top">
                  <div className="cert-left">
                    <span className="cert-level">{level}</span>
                    <span className="cert-title">{title}</span>
                    <span className="cert-issuer">Deloitte USI</span>
                  </div>
                  <div className="cert-right">
                    <div className="cert-dates">
                      <div className="cert-date-item">
                        <span className="cert-date-label">AWARDED</span>
                        <span className="cert-date-val">{year}</span>
                      </div>
                    </div>
                  </div>
                </div>
                <p className="award-quote">&ldquo;{quote}&rdquo;</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

const SECTIONS = {
  skills:         TechStack,
  experience:     Experience,
  education:      Education,
  certifications: Certifications,
  awards:         Awards,
}

function App() {
  const [active, setActive] = useState('home')
  const Section = SECTIONS[active]

  return (
    <div className="page">
      <div className="panel-grid" />

      <main className="hero-panel">
        <nav className="chapter-nav">
          {NAV.map(({ id, Icon, label }) => (
            <button
              key={id}
              className={`chapter-nav-btn${active === id ? ' chapter-nav-btn--active' : ''}`}
              onClick={() => setActive(id)}
            >
              <Icon className="chapter-icon" />
              <span className="chapter-label">{label}</span>
            </button>
          ))}
        </nav>

        <div className="caption-box caption-box--top">
          {CAPTIONS[active]}
        </div>

        {active === 'home' && (
          <div className="hero-layout">

            <div className="photo-panel">
              <div className="photo-frame">
                <img src={profilePlaceholder} alt="Mayur G D" className="profile-photo" />
                <div className="photo-label">MAYUR G D</div>
              </div>
              <div className="badge">
                <span className="badge-role">SENIOR ML ENGINEER</span>
              </div>
              <div className="hero-social">
                <a href="https://www.linkedin.com/in/mayur-g-d-a78a1a131/" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="LinkedIn"><FaLinkedin /></a>
                <a href="https://github.com/mayurgd" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="GitHub"><FaGithub /></a>
                <a href="https://leetcode.com/u/mayur_356/" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="LeetCode"><SiLeetcode /></a>
                <a href={RESUME_URL} download={RESUME_FILENAME} className="hero-social-link" aria-label="Download résumé" title="Download résumé"><FaFileDownload /></a>
              </div>
            </div>

            <div className="info-panel">

              <div className="name-block">
                <div className="name-tag">SENIOR ML ENGINEER &nbsp;·&nbsp; WARNER BROS. DISCOVERY</div>
                <h1 className="hero-name">MAYUR <span className="name-accent">G D</span></h1>
              </div>

              <div className="hero-stats">
                <div className="hero-stat-chip"><span className="hero-stat-val">{yearsSince(CAREER_START)}+</span><span className="hero-stat-label">YEARS IN ML</span></div>
                <div className="hero-stat-chip"><span className="hero-stat-val">{AWARDS.length}</span><span className="hero-stat-label">PERFORMANCE AWARDS</span></div>
                <div className="hero-stat-chip"><span className="hero-stat-val">2×</span><span className="hero-stat-label">DATABRICKS CERTIFIED</span></div>
              </div>

              <div className="speech-bubble">
                <p className="bio-text">
                  Machine learning engineer with {yearsSince(CAREER_START)}+ years of experience taking
                  models from prototype to production — forecasting, clustering, and LLM and agentic
                  AI systems — on Databricks, PySpark and the cloud.
                </p>
                <p className="bio-text bio-text--second">
                  Previously a data scientist at Deloitte USI, where I cut pipeline runtimes by over
                  10x, led a PySpark and MLflow migration, and was recognised with four performance
                  awards. Now building generative AI for audio description at Warner Bros. Discovery.
                </p>
              </div>

              <div className="contact-panel">
                <div className="contact-label">CONTACT</div>
                <div className="contact-items">
                  <div className="contact-item">
                    <span className="contact-icon"><FaEnvelope /></span>
                    <a href="mailto:mayur.dhage356@gmail.com" className="contact-link">mayur.dhage356@gmail.com</a>
                  </div>
                  <div className="contact-item">
                    <span className="contact-icon"><FaLinkedin /></span>
                    <a href="https://www.linkedin.com/in/mayur-g-d-a78a1a131/" target="_blank" rel="noopener noreferrer" className="contact-link">LinkedIn</a>
                  </div>
                  <div className="contact-item">
                    <span className="contact-icon"><FaFileDownload /></span>
                    <a href={RESUME_URL} download={RESUME_FILENAME} className="contact-link">Download résumé</a>
                  </div>                </div>
              </div>

            </div>
          </div>
        )}

        {Section && (
          <div className="section-body section-body--wide">
            <Section />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
