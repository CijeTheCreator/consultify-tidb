import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const specialty = searchParams.get('specialty')

    const whereClause: any = {
      type: 'DOCTOR'
    }

    // Add specialty filter if provided
    if (specialty) {
      whereClause.specialty = specialty
    }

    const doctors = await prisma.user.findMany({
      where: whereClause,
      select: {
        id: true,
        type: true,
        specialty: true,
        language: true,
        createdAt: true,
        updatedAt: true,
      },
      orderBy: {
        createdAt: 'desc'
      }
    })

    return NextResponse.json(doctors)
  } catch (error) {
    console.error('Error fetching doctors:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}