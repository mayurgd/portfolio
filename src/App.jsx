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
import backstageIcon from './assets/icons/backstage.png'
import deloitteIcon from './assets/icons/deloitte.png'
import {
  FaPhoneAlt, FaEnvelope, FaMapMarkerAlt, FaLinkedin, FaGithub,
  FaUser, FaBolt, FaBriefcase, FaFolderOpen, FaGraduationCap,
  FaCertificate, FaTrophy, FaAws,
} from 'react-icons/fa'
import { SiLeetcode, SiPython, SiApachespark, SiGooglecloud, SiDatabricks, SiKubernetes, SiRedhatopenshift, SiFastapi, SiMlflow, SiPytorch, SiTensorflow, SiDocker, SiRedis, SiGithubactions, SiHelm, SiArgo, SiScikitlearn } from 'react-icons/si'

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
      { name: 'Databricks', Icon: SiDatabricks },
      { name: 'Azure', iconSrc: azureIcon },
      { name: 'Kubernetes', Icon: SiKubernetes },
      { name: 'OpenShift',  Icon: SiRedhatopenshift },
      { name: 'VertexAI', iconSrc: vertexaiIcon },
      { name: 'AWS',        Icon: FaAws },
    ],
  },
  {
    category: 'AI / ML',
    accent: '#ef4444',
    chipStyle: true,
    items: [
      { name: 'Supervised & Unsupervised' },
      { name: 'Deep Learning' },
      { name: 'Time Series' },
      { name: 'NLP' },
      { name: 'Gen AI' },
      { name: 'Agentic AI' },
      { name: 'RAG' },
      { name: 'LLM' },
      { name: 'MCP' },
    ],
  },
  {
    category: 'Frameworks & Libraries',
    accent: '#22c55e',
    items: [
      { name: 'FastAPI',     Icon: SiFastapi },
      { name: 'Langfuse', iconSrc: langfuseIcon },
      { name: 'MLFlow',      Icon: SiMlflow },
      { name: 'PyTorch',     Icon: SiPytorch },
      { name: 'TensorFlow',  Icon: SiTensorflow },
      { name: 'scikit-learn', Icon: SiScikitlearn },
    ],
  },
  {
    category: 'Agent Frameworks',
    accent: '#a855f7',
    items: [
      { name: 'LangChain', iconSrc: langchainIcon },
      { name: 'LangGraph', iconSrc: langgraphIcon },
      { name: 'CrewAI', iconSrc: crewaiIcon },
      { name: 'MS Agent Framework', iconSrc: microsoftIcon },
    ],
  },
  {
    category: 'Data & DevOps Tools',
    accent: '#f97316',
    items: [
      { name: 'Docker',         Icon: SiDocker },
      { name: 'Redis',          Icon: SiRedis },
      { name: 'GitHub Actions', Icon: SiGithubactions },
      { name: 'Helm',           Icon: SiHelm },
      { name: 'Argo',           Icon: SiArgo },
      { name: 'Backstage', iconSrc: backstageIcon },
    ],
  },
]

const NAV = [
  { id: 'home',           Icon: FaUser,          label: 'Home'         },
  { id: 'skills',         Icon: FaBolt,          label: 'Tech Stack'   },
  { id: 'experience',     Icon: FaBriefcase,     label: 'Experience'   },
  { id: 'projects',       Icon: FaFolderOpen,    label: 'Projects'     },
  { id: 'education',      Icon: FaGraduationCap, label: 'Education'    },
  { id: 'certifications', Icon: FaCertificate,   label: 'Certifications'},
  { id: 'awards',         Icon: FaTrophy,        label: 'Awards'       },
]

const CAPTIONS = {
  home:           'MEANWHILE... IN THE DATA DIMENSION',
  skills:         'WEAPONS EQUIPPED ⚡ ARSENAL UNLOCKED',
  experience:     'YEARS IN THE FIELD ◆ BATTLE HARDENED',
  projects:       'MISSIONS DEPLOYED ★ CODE IN THE WILD',
  education:      'ORIGIN STORY ✦ WHERE IT ALL BEGAN',
  certifications: 'BADGES OF HONOUR ◈ SKILLS CERTIFIED',
  awards:         'HALL OF FAME 🕷 VICTORIES LOGGED',
}

/* ── Tech Stack section ── */
function TechStack() {
  return (
    <div className="techstack-wrapper">
      <div className="techstack">
        {TECH_STACK.map(({ category, accent, chipStyle, items }, idx) => (
          <div key={category} className="techstack-col" style={{ '--cat-accent': accent }}>
            <div className="techstack-col-header">
              <span className="techstack-col-header-text">{category}</span>
              <span className="techstack-col-num">#{String(idx + 1).padStart(2, '0')}</span>
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
function Experience() {
  const totalStart = new Date('2020-08-17')
  const totalEnd   = new Date('2026-04-12')
  const totalMs    = totalEnd - totalStart
  const totalYears = Math.floor(totalMs / (1000 * 60 * 60 * 24 * 365))
  const totalMonths = Math.floor((totalMs / (1000 * 60 * 60 * 24 * 30.44)) % 12)

  const roles = [
    {
      title: 'Consultant — Data Scientist',
      subtitle: 'Senior Operative',
      period: '01 Jun 2023 — 12 Apr 2026',
      from: new Date('2023-06-01'),
      to:   new Date('2026-04-12'),
      accent: '#3b82f6',
      issue: '02',
      action: 'LEVEL UP!',
    },
    {
      title: 'Analyst — Data Scientist',
      subtitle: 'Origin Arc',
      period: '17 Aug 2020 — 31 May 2023',
      from: new Date('2020-08-17'),
      to:   new Date('2023-05-31'),
      accent: '#22c55e',
      issue: '01',
      action: 'BOOM!',
    },
  ]

  return (
    <div className="exp-wrapper">

      {/* ── Splash header ── */}
      <div className="exp-splash">
        <div className="exp-splash-bg" />
        <div className="exp-splash-left">
          <div className="exp-splash-eyebrow">FIELD REPORT &nbsp;///&nbsp; ACTIVE SINCE 2020</div>
          <div className="exp-splash-company">DELOITTE <span className="exp-splash-usi">USI</span></div>
          <div className="exp-splash-tagline">&ldquo; Turning data into decisions, one sprint at a time. &rdquo;</div>
        </div>
        <div className="exp-splash-center">
          <img src={deloitteIcon} alt="Deloitte" className="exp-splash-logo" />
        </div>
        <div className="exp-splash-right">
          <div className="exp-splash-stat">
            <span className="exp-splash-stat-val">{totalYears}</span>
            <span className="exp-splash-stat-unit">YRS</span>
          </div>
          <div className="exp-splash-divider" />
          <div className="exp-splash-stat">
            <span className="exp-splash-stat-val">{totalMonths}</span>
            <span className="exp-splash-stat-unit">MOS</span>
          </div>
          <div className="exp-splash-stat-label">TOTAL<br/>EXPERIENCE</div>
        </div>
      </div>

      {/* ── Role panels ── */}
      <div className="exp-panels">
        {roles.map(({ title, subtitle, period, from, to, accent, issue, action }, i) => {
          const durMs = to - from
          const durYrs = Math.floor(durMs / (1000 * 60 * 60 * 24 * 365))
          const durMos = Math.floor((durMs / (1000 * 60 * 60 * 24 * 30.44)) % 12)
          const pct = Math.round((durMs / totalMs) * 100)
          return (
            <div key={title} className="exp-timeline-row">
              <div className="exp-timeline-spine">
                <div className="exp-timeline-dot" style={{ background: accent, borderColor: accent }} />
                {i < roles.length - 1 && <div className="exp-timeline-line" style={{ background: `linear-gradient(to bottom, ${accent}, var(--grey-light))` }} />}
              </div>
              <div className="exp-panel" style={{ '--role-accent': accent }}>
                <div className="exp-panel-header">
                  <span className="exp-panel-issue">ISSUE #{issue}</span>
                  <span className="exp-panel-action">{action}</span>
                </div>
                <div className="exp-panel-title">{title}</div>
                <div className="exp-panel-subtitle">{subtitle}</div>
                <div className="exp-panel-period">{period}</div>
                <div className="exp-panel-bar-wrap">
                  <div className="exp-panel-bar" style={{ width: `${pct}%` }} />
                </div>
                <span className="exp-panel-bar-label">{pct}% OF CAREER</span>
                <div className="exp-panel-dur">
                  {durYrs > 0 && <><span className="exp-panel-dur-val">{durYrs}</span><span className="exp-panel-dur-unit">yr</span></>}
                  {durMos > 0 && <><span className="exp-panel-dur-val">{durMos}</span><span className="exp-panel-dur-unit">mo</span></>}
                </div>
              </div>
            </div>
          )
        })}
      </div>
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
    tag: 'UNDERGRADUATE',
    stats: [
      { label: 'GPA', value: '9.24' },
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
      { label: 'GPA', value: '10.0' },
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
                <span className="edu-index">#{String(idx + 1).padStart(2, '0')}</span>
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
      <div className="edu-scroll-tag">KNOWLEDGE ARC — ORIGIN STORY</div>
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
    icon: '🏅',
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
    icon: '⚡',
    level: 'ASSOCIATE',
  },
  {
    id: 'ml-spec',
    title: 'Machine Learning Specialization',
    issuer: 'DeepLearning.AI · Stanford University',
    issued: 'Apr 2024',
    expires: null,
    accent: '#3b82f6',
    tag: 'LIFETIME',
    icon: '🎓',
    level: 'SPECIALIZATION',
  },
  {
    id: 'dsa',
    title: 'Data Structures & Algorithms in Python',
    issuer: 'Coding Ninjas',
    issued: 'Mar 2020',
    expires: null,
    accent: '#22c55e',
    tag: 'LIFETIME',
    icon: '🧩',
    level: 'COMPLETION',
  },
]

function Certifications() {
  return (
    <div className="cert-wrapper">
      <div className="cert-grid">
        {CERTS.map(({ id, title, issuer, issued, expires, accent, tag, icon, level }, idx) => (
          <div key={id} className="cert-card" style={{ '--cert-accent': accent }}>
            <div className="cert-card-top">
              <span className="cert-index">#{String(idx + 1).padStart(2, '0')}</span>
              <span className="cert-tag">{tag}</span>
            </div>
            <div className="cert-icon-wrap">
              <span className="cert-icon">{icon}</span>
            </div>
            <div className="cert-level">{level}</div>
            <div className="cert-title">{title}</div>
            <div className="cert-issuer">{issuer}</div>
            <div className="cert-dates">
              <div className="cert-date-item">
                <span className="cert-date-label">ISSUED</span>
                <span className="cert-date-val">{issued}</span>
              </div>
              {expires ? (
                <div className="cert-date-item">
                  <span className="cert-date-label">EXPIRES</span>
                  <span className="cert-date-val">{expires}</span>
                </div>
              ) : (
                <div className="cert-date-item">
                  <span className="cert-date-label">EXPIRES</span>
                  <span className="cert-date-val cert-date-never">NEVER ✦</span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ── PLACEHOLDER panel for upcoming sections ── */
function ComingSoon({ label }) {
  return (
    <div className="coming-soon">
      <span className="coming-soon-bang">!</span>
      <span className="coming-soon-title">CHAPTER LOADING...</span>
      <span className="coming-soon-sub">{label} — coming soon</span>
    </div>
  )
}

function App() {
  const [active, setActive] = useState('home')

  return (
    <div className="page">
      <div className="halftone-overlay" />
      <div className="panel-grid" />

      <main className="hero-panel">
        {/* Corner tags */}
        <span className="corner-tag">ISSUE #001</span>
        <span className="corner-tag corner-tag--right">2026</span>

        {/* ── Chapter nav tabs ── */}
        <nav className="chapter-nav">
          {NAV.map(({ id, Icon, label }, i) => (
            <button
              key={id}
              className={`chapter-nav-btn${active === id ? ' chapter-nav-btn--active' : ''}`}
              onClick={() => setActive(id)}
            >
              <span className="chapter-num">#{String(i + 1).padStart(2, '0')}</span>
              <Icon className="chapter-icon" />
              <span className="chapter-label">{label}</span>
            </button>
          ))}
        </nav>

        {/* Action caption */}
        <div className="caption-box caption-box--top">
          {CAPTIONS[active]}
        </div>

        {/* ── Section content ── */}
        {active === 'home' && (
          <div className="hero-layout">

            {/* ── Left: Photo panel ── */}
            <div className="photo-panel">
              <div className="photo-frame">
                <img src={profilePlaceholder} alt="Mayur G D" className="profile-photo" />
                <div className="photo-label">MAYUR G D</div>
                <div className="photo-corner photo-corner--tl">✦</div>
                <div className="photo-corner photo-corner--br">✦</div>
              </div>
              <div className="badge">
                <span className="badge-role">DATA SCIENTIST</span>
                <span className="badge-sep">✦</span>
                <span className="badge-role">ML ENGINEER</span>
              </div>
              <div className="hero-social">
                <a href="https://www.linkedin.com/in/mayur-g-d-a78a1a131/" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="LinkedIn"><FaLinkedin /></a>
                <a href="https://github.com/mayurgd" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="GitHub"><FaGithub /></a>
                <a href="https://leetcode.com/u/mayur_356/" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="LeetCode"><SiLeetcode /></a>
              </div>
            </div>

            {/* ── Right: Info panel ── */}
            <div className="info-panel">

              {/* Identity block */}
              <div className="name-block">
                <div className="name-tag">⚡ REAL NAME &nbsp;///&nbsp; DATA OPERATIVE</div>
                <h1 className="hero-name">MAYUR <span className="name-accent">G D</span></h1>
                <div className="hero-alias">[ ALIAS: THE DATA SPIDER ]</div>
              </div>

              {/* Stat chips */}
              <div className="hero-stats">
                <div className="hero-stat-chip"><span className="hero-stat-val">5+</span><span className="hero-stat-label">YRS EXP</span></div>
                <div className="hero-stat-chip"><span className="hero-stat-val">15+</span><span className="hero-stat-label">TECH STACK</span></div>
                <div className="hero-stat-chip"><span className="hero-stat-val">GEN AI</span><span className="hero-stat-label">SPECIALIST</span></div>
                <div className="hero-stat-chip"><span className="hero-stat-val">BLR</span><span className="hero-stat-label">BASE</span></div>
              </div>

              {/* Speech bubble */}
              <div className="speech-bubble">
                <p className="bio-text">
                  I do my best work when there&apos;s a clear direction and something
                  challenging to achieve. I like thinking things through, moving fast, and
                  finding ways to improve along the way.
                </p>
                <p className="bio-text bio-text--second">
                  Outside of work, I focus on discipline - fitness, finance, or
                  learning new skills; because getting better in one area makes
                  everything else easier.
                </p>
              </div>

              {/* Contact dossier */}
              <div className="contact-panel">
                <div className="contact-label">◆ FIELD CONTACT DOSSIER ◆</div>
                <div className="contact-items">
                  <div className="contact-item">
                    <span className="contact-icon"><FaPhoneAlt /></span>
                    <span>+91-9483996212</span>
                  </div>
                  <div className="contact-item">
                    <span className="contact-icon"><FaEnvelope /></span>
                    <a href="mailto:mayur.dhage356@gmail.com" className="contact-link">mayur.dhage356@gmail.com</a>
                  </div>
                  <div className="contact-item">
                    <span className="contact-icon"><FaMapMarkerAlt /></span>
                    <span>Bengaluru, India</span>
                  </div>
                </div>
              </div>

            </div>
          </div>
        )}

        {active === 'skills' && (
          <div className="section-body section-body--wide">
            <TechStack />
          </div>
        )}

        {active === 'experience' && (
          <div className="section-body section-body--wide">
            <Experience />
          </div>
        )}

        {active === 'education' && (
          <div className="section-body section-body--wide">
            <Education />
          </div>
        )}

        {active === 'certifications' && (
          <div className="section-body section-body--wide">
            <Certifications />
          </div>
        )}

        {active !== 'home' && active !== 'skills' && active !== 'experience' && active !== 'education' && active !== 'certifications' && (
          <div className="section-body">
            <ComingSoon label={NAV.find(n => n.id === active).label} />
          </div>
        )}

        {/* Bottom caption */}
        <div className="caption-box caption-box--bottom">
          &ldquo; WITH GREAT DATA COMES GREAT RESPONSIBILITY &rdquo;
        </div>
      </main>

      <div className="action-word">POW!</div>
    </div>
  )
}

export default App
