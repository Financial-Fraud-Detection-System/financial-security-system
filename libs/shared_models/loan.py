from pydantic import BaseModel, Field


class Loan(BaseModel):
    """
    Financial profile and loan-related data of a customer.
    """

    Age: int = Field(..., description="Age of the customer")
    Occupation: int = Field(..., description="Encoded occupation category")
    Annual_Income: float = Field(
        ..., description="Annual income of the customer in USD"
    )
    Num_Bank_Accounts: int = Field(..., description="Number of bank accounts held")
    Num_Credit_Card: int = Field(..., description="Number of credit cards owned")
    Interest_Rate: float = Field(
        ..., description="Interest rate applicable to the customer"
    )
    Num_of_Loan: int = Field(..., description="Total number of loans taken")
    Num_of_Delayed_Payment: int = Field(
        ..., description="Number of delayed payments recorded"
    )

    # One-hot encoded loan types
    Auto_Loan: int = Field(..., alias="Auto Loan")
    Credit_Builder_Loan: int = Field(..., alias="Credit-Builder Loan")
    Debt_Consolidation_Loan: int = Field(..., alias="Debt Consolidation Loan")
    Home_Equity_Loan: int = Field(..., alias="Home Equity Loan")
    Mortgage_Loan: int = Field(..., alias="Mortgage Loan")
    Not_Specified: int = Field(..., alias="Not Specified")
    Payday_Loan: int = Field(..., alias="Payday Loan")
    Personal_Loan: int = Field(..., alias="Personal Loan")
    Student_Loan: int = Field(..., alias="Student Loan")

    model_config = {
        "populate_by_name": True,
        "extra": "forbid",  # optional: to disallow unexpected fields
    }
