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
      { name: 'OpenShift',      Icon: SiRedhatopenshift },
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
  const roles = [
    {
      title: 'Consultant',
      period: '01 Jun 2023 — 12 Apr 2026',
      from: '2023-06-01',
      to: '2026-04-12',
    },
    {
      title: 'Analyst',
      period: '17 Aug 2020 — 31 May 2023',
      from: '2020-08-17',
      to: '2023-05-31',
    },
  ]

  const totalStart = new Date('2020-08-17')
  const totalEnd   = new Date('2026-04-12')
  const totalMs    = totalEnd - totalStart
  const totalYears = Math.floor(totalMs / (1000 * 60 * 60 * 24 * 365))
  const totalMonths = Math.floor((totalMs / (1000 * 60 * 60 * 24 * 30.44)) % 12)

  return (
    <div className="exp-wrapper">
      <div className="exp-company-card">
        <img src={deloitteIcon} alt="Deloitte" className="exp-company-logo" />
        <div className="exp-company-info">
          <div className="exp-company-name">Deloitte USI</div>
          <div className="exp-company-tenure">
            {totalYears} yr {totalMonths} mo &nbsp;·&nbsp; Aug 2020 — Apr 2026
          </div>
        </div>
        <div className="exp-total-badge">
          <span className="exp-total-num">{totalYears}<span className="exp-total-unit">yr</span>{totalMonths}<span className="exp-total-unit">mo</span></span>
          <span className="exp-total-label">TOTAL EXP</span>
        </div>
      </div>

      <div className="exp-timeline">
        {roles.map(({ title, period }, i) => (
          <div key={title} className="exp-role-row">
            <div className="exp-role-connector">
              <div className="exp-role-dot" />
              {i < roles.length - 1 && <div className="exp-role-line" />}
            </div>
            <div className="exp-role-card">
              <div className="exp-role-num">#{String(i + 1).padStart(2, '0')}</div>
              <div className="exp-role-title">{title}</div>
              <div className="exp-role-period">{period}</div>
            </div>
          </div>
        ))}
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
          MEANWHILE... IN THE DATA DIMENSION
        </div>

        {/* ── Section content ── */}
        {active === 'home' && (
          <div className="hero-layout">
            <div className="photo-panel">
              <div className="photo-frame">
                <img src={profilePlaceholder} alt="Mayur G D" className="profile-photo" />
                <div className="photo-label">MAYUR G D</div>
              </div>
              <div className="badge">DATA SCIENTIST<br />& ML ENGINEER</div>
            </div>

            <div className="info-panel">
              <div className="name-block">
                <div className="name-tag">Mr</div>
                <h1 className="hero-name">MAYUR <span className="name-accent">G D</span></h1>
              </div>

              <div className="speech-bubble">
                <p className="bio-text">
                  I do my best work when there&apos;s a clear direction and something
                  challenging to achieve. I like thinking things through, moving fast, and
                  finding ways to improve along the way.
                </p>
                <p className="bio-text bio-text--second">
                  Outside of work, I focus on discipline - whether it&apos;s fitness,
                  finance, or learning new skills - because
                  getting better in one area makes everything else easier.
                </p>
              </div>

              <div className="contact-panel">
                <div className="contact-label">— CONTACT —</div>
                <div className="contact-items">
                  <div className="contact-item">
                    <span className="contact-icon"><FaPhoneAlt /></span>
                    <span>9483996212</span>
                  </div>
                  <div className="contact-item">
                    <span className="contact-icon"><FaEnvelope /></span>
                    <a href="mailto:mayur.dhage356@gmail.com" className="contact-link">
                      mayur.dhage356@gmail.com
                    </a>
                  </div>
                  <div className="contact-item">
                    <span className="contact-icon"><FaMapMarkerAlt /></span>
                    <span>Bengaluru, India</span>
                  </div>
                </div>
              </div>

            </div>

            <div className="hero-social">
              <a href="https://www.linkedin.com/in/mayur-g-d-a78a1a131/" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="LinkedIn">
                <FaLinkedin />
              </a>
              <a href="https://github.com/mayurgd" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="GitHub">
                <FaGithub />
              </a>
              <a href="https://leetcode.com/u/mayur_356/" target="_blank" rel="noopener noreferrer" className="hero-social-link" aria-label="LeetCode">
                <SiLeetcode />
              </a>
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

        {active !== 'home' && active !== 'skills' && active !== 'experience' && active !== 'education' && (
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
