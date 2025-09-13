"use client"

import { useState, useEffect } from "react"
import { useAuth } from "@/contexts/auth-context"
import Navbar from "./navbar"
import type { User } from "@/lib/types"

export default function NavbarWrapper() {
  const [dbUser, setDbUser] = useState<User | null>(null)
  const { user: supabaseUser, signOut, loading } = useAuth()

  useEffect(() => {
    const fetchUserProfile = async () => {
      if (!supabaseUser?.email) {
        setDbUser(null)
        return
      }

      try {
        const response = await fetch(`/api/users?email=${encodeURIComponent(supabaseUser.email)}`)
        if (response.ok) {
          const userData = await response.json()
          setDbUser(userData)
        } else {
          setDbUser(null)
        }
      } catch (error) {
        console.error('Error fetching user profile:', error)
        setDbUser(null)
      }
    }

    fetchUserProfile()
  }, [supabaseUser])

  const handleSignOut = async () => {
    await signOut()
    setDbUser(null)
  }

  // Show loading state or return navbar with user data
  if (loading) {
    return <Navbar user={null} onSignOut={handleSignOut} />
  }

  return <Navbar user={dbUser} onSignOut={handleSignOut} />
}