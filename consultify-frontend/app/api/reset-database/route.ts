import { NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function DELETE() {
  try {
    // Delete all records in the correct order to avoid foreign key constraints
    await prisma.message.deleteMany({})
    await prisma.prescription.deleteMany({})
    await prisma.consultation.deleteMany({})
    await prisma.user.deleteMany({})

    return NextResponse.json({ 
      message: 'Database reset successfully',
      deletedRecords: {
        messages: 'all',
        prescriptions: 'all',
        consultations: 'all',
        users: 'all'
      }
    })
  } catch (error) {
    console.error('Error resetting database:', error)
    return NextResponse.json(
      { error: 'Failed to reset database' },
      { status: 500 }
    )
  } finally {
    await prisma.$disconnect()
  }
}