from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Tuple, TYPE_CHECKING

from llvmlite import ir


@dataclass
class Type:
    def get_ir_type(self) -> ir.Type:
        raise NotImplementedError(f'Not implemented for {type(self)}')

    @property
    def name(self) -> str:
        raise NotImplementedError(f'Not implemented for {type(self)}')

    def as_pointer(self) -> PointerType:
        return PointerType(self)

    def as_pointee(self) -> Type:
        assert isinstance(self, PointerType)
        return self.pointee

    def __eq__(self, other: Any) -> bool:
        return hasattr(other, 'name') and self.name == other.name

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        assert self == from_type, f'Cannot adapt {from_type.name} to {self.name}'
        return value


@dataclass
class VoidType(Type):
    @property
    def name(self) -> str:
        return 'void'

    def get_ir_type(self) -> ir.Type:
        return ir.VoidType()


void = VoidType()


@dataclass
class IntType(Type):
    bits: int
    signed: bool

    @property
    def name(self) -> str:
        prefix = 'i' if self.signed else 'u'
        return f'{prefix}{self.bits}'

    def get_ir_type(self) -> ir.Type:
        return ir.IntType(self.bits)

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        if not isinstance(from_type, IntType):
            return super().adapt(builder, value, from_type)

        if from_type.bits == self.bits:
            assert from_type.signed == self.signed
            return value

        ir_type = self.get_ir_type()
        if from_type.bits > self.bits:
            return builder.trunc(value, ir_type)

        if from_type.signed:
            assert self.signed
            return builder.sext(value, ir_type)
        else:
            assert not self.signed
            return builder.zext(value, ir_type)


@dataclass
class FloatType(Type):
    bits: int

    @property
    def name(self) -> str:
        return f'f{self.bits}'

    def get_ir_type(self) -> ir.Type:
        return ir.FloatType() if self.bits == 32 else ir.DoubleType()


@dataclass
class BoolType(Type):
    @property
    def name(self) -> str:
        return 'bool'

    def get_ir_type(self) -> ir.Type:
        return ir.IntType(1)


@dataclass
class PointerType(Type):
    pointee: Type

    @property
    def name(self) -> str:
        return self.pointee.name + '*'

    def get_ir_type(self) -> ir.Type:
        return self.pointee.get_ir_type().as_pointer()

    def adapt(self, builder: ir.IRBuilder, value: ir.Value, from_type: Type) -> ir.Value:
        # TODO remove this
        i64 = ir.IntType(64)

        if isinstance(from_type, ArrayType) and self.pointee == from_type.element_type:
            memory = builder.alloca(value.type)
            builder.store(value, memory)
            return builder.gep(memory, (i64(0), i64(0)))

        return super().adapt(builder, value, from_type)


@dataclass
class StructType(Type):
    struct_name: str
    members: List[Tuple[str, Type]]

    @property
    def name(self) -> str:
        return self.struct_name

    def get_ir_type(self) -> ir.Type:
        member_types = [t.get_ir_type() for n, t in self.members]
        return ir.LiteralStructType(member_types)

    def get_member_index(self, name: str) -> int:
        for i, (n, t) in enumerate(self.members):
            if n == name:
                return i
        raise KeyError()

    def get_member_type(self, name: str) -> Type:
        for n, t in self.members:
            if n == name:
                return t
        raise KeyError()


@dataclass
class ArrayType(Type):
    element_type: Type
    length: int

    @property
    def name(self) -> str:
        return f'{self.element_type.name}[{self.length}]'

    def get_ir_type(self) -> ir.Type:
        return ir.ArrayType(self.element_type.get_ir_type(), self.length)


@dataclass
class FunctionType(Type):
    parameter_types: List[Type]
    return_type: Type
    variadic: bool

    def get_ir_type(self) -> ir.Type:
        return ir.FunctionType(
            self.return_type.get_ir_type(),
            [t.get_ir_type() for t in self.parameter_types],
            self.variadic,
        )
