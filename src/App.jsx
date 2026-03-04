import { useState } from 'react'
import './App.css'
import profilePlaceholder from './assets/my_image_1.png'
import {
  FaPhoneAlt, FaEnvelope, FaMapMarkerAlt, FaLinkedin, FaGithub,
  FaUser, FaBolt, FaBriefcase, FaFolderOpen, FaGraduationCap,
  FaCertificate, FaTrophy,
} from 'react-icons/fa'
import { SiLeetcode } from 'react-icons/si'

const NAV = [
  { id: 'home',           Icon: FaUser,          label: 'Home'         },
  { id: 'skills',         Icon: FaBolt,          label: 'Skills'       },
  { id: 'experience',     Icon: FaBriefcase,     label: 'Experience'   },
  { id: 'projects',       Icon: FaFolderOpen,    label: 'Projects'     },
  { id: 'education',      Icon: FaGraduationCap, label: 'Education'    },
  { id: 'certifications', Icon: FaCertificate,   label: 'Certifications'},
  { id: 'awards',         Icon: FaTrophy,        label: 'Awards'       },
]

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





      {/* ── Left nav strip ── */}
      <div className="nav-strip">
        {NAV.map(({ id, Icon, label }) => (
          <button
            key={id}
            className={`nav-strip-btn${active === id ? ' nav-strip-btn--active' : ''}`}
            onClick={() => setActive(id)}
            aria-label={label}
          >
            <Icon className="nav-strip-icon" />
            <span className="nav-strip-label">{label}</span>
          </button>
        ))}
      </div>

      <main className="hero-panel">
        {/* Corner tags */}
        <span className="corner-tag">ISSUE #001</span>
        <span className="corner-tag corner-tag--right">2026</span>

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

        {active !== 'home' && (
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
