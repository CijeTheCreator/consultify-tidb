"use client"

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useAuth } from '@/contexts/auth-context'
import OnboardingFlow from '@/components/onboarding/onboarding-flow'

export default function AuthSuccess() {
  const { user, loading } = useAuth()
  const router = useRouter()
  const [showOnboarding, setShowOnboarding] = useState(false)
  const [checkingUser, setCheckingUser] = useState(true)

  useEffect(() => {
    if (!loading) {
      if (user) {
        checkUserExists()
      } else {
        router.push('/')
      }
    }
  }, [user, loading, router])

  const checkUserExists = async () => {
    try {
      if (!user?.email) {
        setShowOnboarding(true)
        setCheckingUser(false)
        return
      }

      // Check if user exists in database
      const response = await fetch(`/api/users?email=${encodeURIComponent(user.email)}`)
      
      if (response.ok) {
        const userData = await response.json()
        // User exists in database, redirect to appropriate dashboard
        if (userData.type === 'DOCTOR') {
          router.push('/doctor-dashboard')
        } else {
          router.push('/patient-dashboard')
        }
      } else if (response.status === 404) {
        // User not found in database, needs onboarding
        setShowOnboarding(true)
      } else {
        throw new Error('Failed to check user status')
      }
    } catch (error) {
      console.error('Error checking user status:', error)
      setShowOnboarding(true)
    } finally {
      setCheckingUser(false)
    }
  }

  const handleOnboardingComplete = () => {
    setShowOnboarding(false)
    // The onboarding flow already handles navigation
  }

  if (loading || checkingUser) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-2 text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  if (showOnboarding) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <OnboardingFlow 
            userEmail={user?.email || ''} 
            onComplete={handleOnboardingComplete} 
          />
        </div>
      </div>
    )
  }

  return null
}