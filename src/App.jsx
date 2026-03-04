import './App.css'
import profilePlaceholder from './assets/my_image_1.png'
import { FaPhoneAlt, FaEnvelope, FaMapMarkerAlt } from 'react-icons/fa'

function App() {
  return (
    <div className="page">
      {/* Halftone dot overlay */}
      <div className="halftone-overlay" />

      {/* Panel grid lines background */}
      <div className="panel-grid" />

      <main className="hero-panel">
        {/* Corner accent */}
        <span className="corner-tag">ISSUE #001</span>
        <span className="corner-tag corner-tag--right">2026</span>

        {/* Action caption */}
        <div className="caption-box caption-box--top">
          MEANWHILE... IN THE DATA DIMENSION
        </div>

        {/* Content layout */}
        <div className="hero-layout">
          {/* Photo panel */}
          <div className="photo-panel">
            <div className="photo-frame">
              <img
                src={profilePlaceholder}
                alt="Mayur G D"
                className="profile-photo"
              />
              <div className="photo-label">MAYUR G D</div>
            </div>
            <div className="badge">DATA SCIENTIST<br />& ML ENGINEER</div>
          </div>

          {/* Info panel */}
          <div className="info-panel">
            {/* Name */}
            <div className="name-block">
              <div className="name-tag">Mr</div>
              <h1 className="hero-name">MAYUR <span className="name-accent">G D</span></h1>
            </div>

            {/* Bio speech bubble */}
            <div className="speech-bubble">
              <p className="bio-text">
                I do my best work when there&apos;s a clear direction and something
                challenging to achieve. I like thinking things through, moving fast, and
                finding ways to improve along the way.
              </p>
              <p className="bio-text bio-text--second">
                Outside of work, I focus on discipline — whether it&apos;s fitness,
                finance, or learning new skills — because
                getting better in one area makes everything else easier.
              </p>
              <p className="bio-text bio-text--second">
                I enjoy tackling complex problems, but I try to keep things
                simple and actionable. For me, it&apos;s not just about reaching goals —
                it&apos;s about building the ability to reach even bigger ones.
              </p>
            </div>

            {/* Contact panel */}
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
        </div>

        {/* Bottom caption */}
        <div className="caption-box caption-box--bottom">
          &ldquo; WITH GREAT DATA COMES GREAT RESPONSIBILITY &rdquo;
        </div>
      </main>

      {/* Comic action word */}
      <div className="action-word">POW!</div>
    </div>
  )
}

export default App
