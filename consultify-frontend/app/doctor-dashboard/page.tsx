"use client"

import { useState, useEffect } from "react"
import { useAuth } from "@/contexts/auth-context"
import DoctorLandingPage from "@/components/doctor-landing-page"
import { useRouter } from "next/navigation"

interface UserData {
  id: string
  name: string
  type: string
  specialty: string
  language: string
  consultationsAsDoctor: any[]
}

interface User {
  id: string
  name: string
  language: string
  specialization: string
}

export default function DoctorDashboard() {
  const { user: authUser, loading: authLoading } = useAuth()
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [userData, setUserData] = useState<UserData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchUserData = async () => {
      if (authLoading) return
      
      if (!authUser) {
        router.push('/hero')
        return
      }

      try {
        // Fetch user data from database using email
        const response = await fetch(`/api/users?email=${authUser.email}`)
        if (!response.ok) {
          throw new Error('User not found in database')
        }
        
        const data = await response.json()
        
        // Verify this is a doctor
        if (data.type !== 'DOCTOR') {
          setError('Access denied: This page is for doctors only')
          return
        }

        setUserData(data)
        setUser({
          id: data.id,
          name: data.name,
          language: data.language,
          specialization: data.specialty || 'General Medicine'
        })
      } catch (err) {
        console.error('Error fetching user data:', err)
        setError(err instanceof Error ? err.message : 'Failed to fetch user data')
      } finally {
        setLoading(false)
      }
    }

    fetchUserData()
  }, [authUser, authLoading, router])


  const handleViewConsultations = () => {
    router.push('/consultations')
  }

  if (authLoading || loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-600 mx-auto mb-4"></div>
          <p>Loading your practice dashboard...</p>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-600 mb-4">Error: {error}</p>
          <button
            onClick={() => router.push('/hero')}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Back to Home
          </button>
        </div>
      </div>
    )
  }

  return (
    <DoctorLandingPage
      user={user}
      onViewConsultations={handleViewConsultations}
    />
  )
}
