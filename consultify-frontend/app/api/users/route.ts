import { NextRequest, NextResponse } from 'next/server'
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

export async function POST(request: NextRequest) {
  try {
    const { email, name, type, specialty, language } = await request.json()

    if (!email || !name || !type || !language) {
      return NextResponse.json(
        { error: 'Email, name, type and language are required' },
        { status: 400 }
      )
    }

    if (type === 'DOCTOR' && !specialty) {
      return NextResponse.json(
        { error: 'Specialty is required for doctors' },
        { status: 400 }
      )
    }

    const user = await prisma.user.create({
      data: {
        email,
        name,
        type,
        specialty: type === 'DOCTOR' ? specialty : null,
        language,
      },
    })

    return NextResponse.json(user)
  } catch (error) {
    console.error('Error creating user:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const id = searchParams.get('id')
    const email = searchParams.get('email')

    if (!id && !email) {
      return NextResponse.json(
        { error: 'User ID or email is required' },
        { status: 400 }
      )
    }

    const whereClause = id ? { id } : { email: email! }

    const user = await prisma.user.findUnique({
      where: whereClause,
      include: {
        consultationsAsPatient: true,
        consultationsAsDoctor: true,
        prescriptions: true,
      },
    })

    if (!user) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      )
    }

    return NextResponse.json(user)
  } catch (error) {
    console.error('Error fetching user:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}