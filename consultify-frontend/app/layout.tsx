import type { Metadata } from 'next'
import './globals.css'
import NavbarWrapper from '@/components/navbar-wrapper'
import { Toaster } from 'sonner'
import { AuthProvider } from '@/contexts/auth-context'

export const metadata: Metadata = {
  title: 'Consultify - Multilingual Doctor Consultations',
  description: 'Connect with doctors, no matter the language. Built on AO.',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>
        <AuthProvider>
          <NavbarWrapper />
          {children}
          <Toaster />
        </AuthProvider>
      </body>
    </html>
  )
}
