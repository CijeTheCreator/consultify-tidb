"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Bell, ArrowLeft, Calendar, AlertCircle } from "lucide-react"
import { motion } from "framer-motion"
import { supabase } from "@/lib/supabase"

interface Notification {
  id: string
  message: string
  timestamp: string
  consultationId?: string
  type: 'system' | 'consultation' | 'prescription'
}

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<Notification[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  useEffect(() => {
    fetchNotifications()
  }, [])

  const fetchNotifications = async () => {
    try {
      setLoading(true)
      setError(null)

      // Get current user
      const { data: { user } } = await supabase.auth.getUser()
      if (!user) {
        setError("Please log in to view notifications")
        return
      }

      // Fetch recent system messages and consultation updates
      // For now, we'll fetch recent messages from user's consultations as notifications
      const response = await fetch('/api/consultations', {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      })

      if (!response.ok) {
        throw new Error('Failed to fetch consultations')
      }

      const consultations = await response.json()
      
      // Create notifications from recent consultation activity
      const notifications: Notification[] = []
      
      for (const consultation of consultations.slice(0, 10)) { // Limit to recent 10
        notifications.push({
          id: consultation.id,
          message: `Consultation ${consultation.state.toLowerCase()} - Check your consultation for updates`,
          timestamp: consultation.updatedAt,
          consultationId: consultation.id,
          type: 'consultation'
        })
      }

      // Sort by timestamp (newest first)
      const sortedNotifications = notifications.sort((a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      )

      setNotifications(sortedNotifications)
    } catch (error) {
      console.error("Error fetching notifications:", error)
      setError("Failed to fetch notifications")
    } finally {
      setLoading(false)
    }
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleString()
  }

  const formatMessage = (message: string) => {
    return message
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-cream to-sage-green/10 pt-20">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center mb-8">
          <Button
            variant="ghost"
            onClick={() => router.back()}
            className="mr-4 text-forest-green hover:bg-sage-green/20"
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          <div className="flex items-center">
            <Bell className="w-8 h-8 text-forest-green mr-3" />
            <h1 className="text-3xl font-bold text-forest-green">Notifications</h1>
          </div>
        </div>

        {/* Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          {loading ? (
            <Card>
              <CardContent className="p-8 text-center">
                <div className="animate-spin w-8 h-8 border-4 border-forest-green border-t-transparent rounded-full mx-auto mb-4"></div>
                <p className="text-sage-green">Loading notifications...</p>
              </CardContent>
            </Card>
          ) : error ? (
            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-8 text-center">
                <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-red-700 mb-2">Error Loading Notifications</h3>
                <p className="text-red-600 mb-4">{error}</p>
                <Button
                  onClick={fetchNotifications}
                  variant="outline"
                  className="border-red-300 text-red-700 hover:bg-red-100"
                >
                  Try Again
                </Button>
              </CardContent>
            </Card>
          ) : notifications.length === 0 ? (
            <Card>
              <CardContent className="p-12 text-center">
                <Bell className="w-16 h-16 text-sage-green/50 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-forest-green mb-2">No Notifications</h3>
                <p className="text-sage-green">You don't have any notifications at the moment.</p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-4">
              {notifications.map((notification, index) => (
                <motion.div
                  key={notification.id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                >
                  <Card className="hover:shadow-lg transition-shadow duration-200">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg text-forest-green">
                          {notification.type === 'consultation' ? 'Consultation Update' : 
                           notification.type === 'prescription' ? 'Prescription Update' : 'System Notification'}
                        </CardTitle>
                        <div className="flex items-center text-sm text-sage-green">
                          <Calendar className="w-4 h-4 mr-1" />
                          {formatTimestamp(notification.timestamp)}
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-forest-green leading-relaxed whitespace-pre-line">
                        {formatMessage(notification.message)}
                      </p>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  )
}
